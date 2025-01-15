# Quantifying the Impact of Emotions in Human-LLM Interaction

This project aims to quantify the impact of emotions in Human-Computer Interaction (HCI) between humans and Large Language Models (LLMs). The central hypothesis is that apathetic prompts lead to better performance across various tasks compared to empathetic prompts.

## Project Milestones

1. **Dataset Selection and Prompt Engineering:**
    * Utilize the following datasets:
        * SuperGLUE: For comparative analysis across various tasks (NER, lexical entailment, etc.) using apathetic and empathetic prompts.
        * EmotionX: To analyze if the emotion expressed in the prompt is reflected in the LLM's response.
        * SQuAD (Stanford Question Answering Dataset): To evaluate text comprehension and information retrieval in apathetic and empathetic contexts.
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
* **EmotionX:** A dataset for analyzing emotions in textual conversations.
* **SQuAD (Stanford Question Answering Dataset):** A reading comprehension dataset consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage.

## References

While the email provided URLs for the datasets, this readme omits them in accordance with the prompt's restrictions. It is assumed readers can readily find these datasets with the provided names.
