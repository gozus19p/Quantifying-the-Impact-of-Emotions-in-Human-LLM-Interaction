# Quantifying Emotional Influence in Human–Large Language Model Interaction: Evidence from an Experimental Framework

This repository contains the code, datasets, and resources used in the experiments presented in the paper:

**Quantifying Emotional Influence in Human–Large Language Model Interaction: Evidence from an Experimental Framework**  
Submitted to *Cognitive Computation (Springer Nature)*.

## Overview

The study investigates how emotional tone influences interactions between humans and Large Language Models (LLMs), and vice versa. It is based on two experiments:

- **Experiment Alpha** – Evaluates how LLM-generated emotions impact human responses.
- **Experiment Omega** – Assesses whether fine-tuned LLMs with emotional bias exhibit behavioral divergence when queried.

## Repository Structure

```
.
├── data/
│   ├── alpha\_dataset.csv         # Sampled and annotated utterances from MELD
│   └── omega\_generated\_data.json # Synthetic conversations (can be regenerated)
├── scripts/
│   ├── alpha\_analysis.py         # Statistical analysis for Experiment Alpha
│   ├── omega\_generation.py       # Prompting and generation pipeline
│   └── omega\_evaluation.py       # Evaluation metrics and comparison logic
├── models/
│   └── lora\_finetuned\_model/     # Weights/config (or link if too large)
├── results/
│   └── alpha\_stats.csv
│   └── omega\_stats.csv
├── figures/
│   └── \*.png                     # Visualizations used in the paper
├── requirements.txt
└── README.md
```

## Reproducing the Experiments

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
````

2. **Set up the environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Experiment Alpha analysis**:

   ```bash
   python scripts/alpha_analysis.py
   ```

4. **Run Experiment Omega (generation and evaluation)**:

   * Fine-tuned model should be placed in `models/lora_finetuned_model/` or modified accordingly in `omega_generation.py`.

   ```bash
   python scripts/omega_generation.py
   python scripts/omega_evaluation.py
   ```

## Data and Model Availability

* The MELD dataset is publicly available [here](https://github.com/declare-lab/MELD). Please ensure compliance with its license.
* Synthetic data used in Experiment Omega can be regenerated using the provided scripts.
* Fine-tuned models may be large and hosted externally (e.g., Hugging Face or Zenodo). See [models/README.md](models/README.md) for access instructions.

## Citation

If you use this work, please cite:

```bibtex
@article{your2025emotionalLLM,
  title={Quantifying Emotional Influence in Human–Large Language Model Interaction: Evidence from an Experimental Framework},
  author={Your Name and Collaborators},
  journal={Cognitive Computation},
  year={2025}
}
```