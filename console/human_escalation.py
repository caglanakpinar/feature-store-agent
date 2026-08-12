"""Human escalation gate: decide whether an agent's output needs a person to look at it.

Every other judge in this repo (`problem_judger`, `data_judger`, ...) is wired through
`agentic_configurations.yaml` and scores one pipeline stage against fixed thresholds. This one is
different on purpose: it runs after *any* agent call, on any output, and its LLM is never one of the
`llms:` entries in either package's config — those are sized and picked for the pipeline they gate,
and a stuck classifier or a burned-out judger shouldn't be told "everything's fine" by the very model
that produced it. `HumanEscalation` is handed its own caller instead, built with `build_llm(...)`
straight from arguments (or `HumanEscalation.default()`, which does that for you).

    from console.human_escalation import HumanEscalation

    escalation = HumanEscalation.default(api_key="CLAUDE")
    verdict = escalation.review(agent, agent_output, question=question, context=context)

    if verdict.needs_human:
        for qa in verdict.questions:
            print(qa.render())   # Q: ... / A: ... / B: ... / C: ... / D: <write your own>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from builder.agents import BaseAgent
from builder.factory import build_llm
from models.llms import BaseLLM
from uilts.logger import logger

DEFAULT_ESCALATION_MODEL = "claude/claude-opus-5"  # deliberately not one of the `judger_agent_*` entries

# The fourth option is never generated — it's the human's own escape hatch, always present so a
# question never forces a choice among only what the model thought to offer.
FREEFORM_OPTION_LABEL = "D"
FREEFORM_OPTION_TEXT = "None of the above — write the input/answer you'd give instead."

_LABEL_RE = re.compile(r"^\s*LABEL\s*:\s*([01])\s*$", re.IGNORECASE)
_REASON_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE)
_QUESTION_RE = re.compile(r"^\s*Q\s*:\s*(.+)$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*([A-C])\s*:\s*(.+)$", re.IGNORECASE)


@dataclass
class EscalationQuestion:
    """One question a human is asked, with lettered answers — `D` is always the freeform escape hatch."""

    question: str
    options: dict[str, str]  # "A" -> answer text, in letter order; "D" is appended by `finalize`

    def finalize(self) -> "EscalationQuestion":
        options = dict(self.options)
        options[FREEFORM_OPTION_LABEL] = FREEFORM_OPTION_TEXT
        return EscalationQuestion(question=self.question, options=options)

    def render(self) -> str:
        lines = [f"Q: {self.question}"]
        lines.extend(f"{letter}: {answer}" for letter, answer in self.options.items())
        return "\n".join(lines)


@dataclass
class EscalationVerdict:
    """The outcome of one `HumanEscalation.review` call."""

    agent_name: str
    label: int  # 0: does not need a human, 1: needs a human
    reason: str
    questions: list[EscalationQuestion] = field(default_factory=list)

    @property
    def needs_human(self) -> bool:
        return self.label == 1

    def render(self) -> str:
        if not self.needs_human:
            return f"[{self.agent_name}] no escalation needed: {self.reason}"

        blocks = [f"[{self.agent_name}] needs a human: {self.reason}", ""]
        blocks.extend(qa.render() + "\n" for qa in self.questions)
        return "\n".join(blocks).rstrip()


class HumanEscalation:
    """Classifies an agent's output as escalate-or-not, then drafts the human's review questions.

    One LLM call does both: the judger reads the agent's output the same way `JudgerAgent` does — it
    only ever sees what the agent actually produced, not the pipeline's internal state — and returns a
    label plus, when the label is 1, a small set of `Q: ... / A: ... / B: ... / C: ...` prompts a human
    can act on directly. `D` is never asked of the model; `EscalationQuestion.finalize` appends it so a
    human is never limited to the three answers the model happened to think of.

    Args:
        llm: The judger caller. Must not be one of this pipeline's configured `llms:` entries — build it
            with `build_llm(configs=None, ...)` or use `HumanEscalation.default()`.
        max_questions: Upper bound on how many `Q:` blocks are kept from one review.
    """

    def __init__(self, llm: BaseLLM, max_questions: int = 3) -> None:
        self.llm = llm
        self.max_questions = max_questions

    @classmethod
    def default(
        cls,
        model_name: str = DEFAULT_ESCALATION_MODEL,
        api_key: str = "CLAUDE",
        max_tokens: int = 4000,
        max_questions: int = 3,
    ) -> "HumanEscalation":
        """Build the judger caller from arguments alone — no `configs`, so no `agentic_configurations.yaml` entry is reused."""
        llm = build_llm(
            configs=None,
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            type="judger",
        )
        return cls(llm, max_questions=max_questions)

    def _prompt(self, agent: BaseAgent, agent_output: str, question: str, context: str) -> str:
        return f"""
You are the human-escalation gate sitting behind the agent named "{agent.name}" (type: {agent.type}).
Read what it produced and decide whether a person needs to look at this before it goes further.

Escalate (label 1) when the output is wrong, contradicts the question, is missing something it should
have covered, admits uncertainty or failure, or asks for something only a human can supply (a decision,
a credential, a judgment call). Do not escalate (label 0) when the output plainly answers the question
and there is nothing for a human to add.

question: {question or "(none given)"}
context: {context or "(none given)"}

--- {agent.name} output ---
{agent_output}
--- end output ---

Respond in exactly this format and nothing else. If LABEL is 0, omit the QUESTIONS section entirely.

LABEL: 0 or 1
REASON: one sentence, specific to what you read above
QUESTIONS:
Q: a question a human reviewer would ask, phrased the way a person hitting this would ask it
A: first candidate answer/resolution
B: second candidate answer/resolution
C: third candidate answer/resolution

Up to {self.max_questions} `Q:`/`A:`/`B:`/`C:` blocks, one per distinct issue you found. Never emit a
`D:` line — that option is added separately.
"""

    def review(
        self,
        agent: BaseAgent,
        agent_output: str,
        question: str = "",
        context: str = "",
        **kwargs: Any,
    ) -> EscalationVerdict:
        """Classify `agent_output` and, when it needs a human, draft the questions to ask one."""
        prompt = self._prompt(agent, agent_output, question, context)
        raw = self.llm._call(prompt, **kwargs)
        verdict = self._parse(agent.name, raw)
        logger.info(
            f"HumanEscalation: {agent.name} -> label={verdict.label} "
            f"({len(verdict.questions)} question(s)); {verdict.reason}"
        )
        return verdict

    def _parse(self, agent_name: str, raw: str) -> EscalationVerdict:
        label = 0
        reason = ""
        questions: list[EscalationQuestion] = []
        current: EscalationQuestion | None = None

        for line in raw.splitlines():
            if match := _LABEL_RE.match(line):
                label = int(match.group(1))
                continue
            if match := _REASON_RE.match(line):
                reason = match.group(1).strip()
                continue
            if match := _QUESTION_RE.match(line):
                if current is not None:
                    questions.append(current.finalize())
                current = EscalationQuestion(question=match.group(1).strip(), options={})
                continue
            if match := _ANSWER_RE.match(line):
                if current is None:
                    logger.warning(f"HumanEscalation: {agent_name} answer line before any question: {line!r}")
                    continue
                current.options[match.group(1).upper()] = match.group(2).strip()
                continue

        if current is not None:
            questions.append(current.finalize())

        if label == 1 and not questions:
            logger.warning(f"HumanEscalation: {agent_name} labeled 1 but produced no questions.")

        return EscalationVerdict(
            agent_name=agent_name,
            label=label,
            reason=reason or "(no reason given)",
            questions=questions[: self.max_questions],
        )
