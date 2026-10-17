# Tool turns

One short prompt per tool-bearing agent, for a small local model. The agents' own prompts
(`data_engineer/prompts/`, `feature_engineering/prompts/`) are two to three thousand tokens of
method; a 0.5B model given one writes about calling its tools instead of calling them (0 calls in 7
turns, measured in `benchmarks/RESULTS.md`), and the same model asked in a sentence calls the right
tool (35 of 35). So a small model gets a sentence here, and the console does the rest of what the
long prompt asks for.

Write each prompt the way a person would ask for the task, and name what the agent works on, not the
tool: the model reads the tool's description and picks. A sentence with `{target}` in it is dropped
when no target is known yet.

- `{path}` is the table the agent works on now: the source for the first agent, the previous stage's
  output after that.
- `{target}` is the column being predicted.

## data_reader_agent

Read the dataset at {path} and tell me what is in it.

## data_analyzer_agent

Analyse the dataset at {path}. The column to predict is {target}.

## data_preprocessor_agent

Clean the dataset at {path}: fix the types and drop duplicate rows.

## data_quality_agent

Check the quality of the dataset at {path}. The column to predict is {target}.

## missing_value_agent

Find out which columns of the dataset at {path} have missing values, then fill them.

## feature_prep_agent

Build features from the dataset at {path}.

## feature_selection_agent

Select the features worth keeping in {path}. The column to predict is {target}.
