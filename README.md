# Quantifying the Impact of Emotions in Human-LLM Interaction

This project aims to quantify the impact of emotions in Human-Computer Interaction (HCI) between humans and Large Language Models (LLMs). The central hypothesis is that apathetic prompts lead to better performance across various tasks compared to empathetic prompts.

## Status

| Task | Status | Notes |
| ---- | ------ | ----- |
| MultiRC | To **NOT** do | Too complicated in terms of number of tokens.    |
| AX-b | To **NOT** do  | This won't be done because the dataset is not uniform and it complicates the evaluation. Anyway, the tasks related to this dataset are all already covered by others.      |
| AX-g | Prepared  |    |
| ReCoRD | Completed | Defined a custom metric called "mean BLEU".     |
| CB   | Completed  |    |
| WSC | Completed |      |
| BoolQ | Completed |   |
| COPA | Completed  |   |
| RTE | Completed |     |
| WiC | Completed |     |


## Project Milestones

1. **Dataset Selection and Prompt Engineering:**
    * Utilize SuperGLUE: For comparative analysis across various tasks (NER, lexical entailment, etc.) using apathetic and empathetic prompts. https://super.gluebenchmark.com/tasks
    * Formulate two pairs of prompts (apathetic/empathetic) for each dataset.

2. **Model Selection and Experiment Execution:**
    * Employ a diverse set of at least ten open-weight LLMs to ensure reproducibility.
    * Set temperature to zero for consistent and replicable results, minimizing hallucinations.
    * Execute experiments by feeding the crafted prompts to the selected LLMs and recording their outputs.

3. **Result Analysis and Hypothesis Validation:**
    * Analyze the collected outputs to identify potential monotonic trends across models. The initial hypothesis is that apathetic prompts yield better results.
    * Employ inductive reasoning to verify the hypothesis.
    * Validate the findings using appropriate statistical methods.

4. **Reporting and Publication:**
    * Document the entire process, including methodology, results, and analysis.
    * Explore potential publication opportunities given the anticipated volume of research material.

## Datasets

* **SuperGLUE:** A benchmark suite for evaluating the performance of language models across a range of tasks.

## References

While the email provided URLs for the datasets, this readme omits them in accordance with the prompt's restrictions. It is assumed readers can readily find these datasets with the provided names.
