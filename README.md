# Quantifying the Impact of Emotions in Human–LLM Interaction

This repository contains the code, pipelines, and experimental resources for the paper:

> **Quantifying Emotional Influence in Human–Large Language Model Interaction: Evidence from an Experimental Framework**
> Manuel Gozzi and Francesca Fallucchi
> Submitted to *Electronics* (MDPI), manuscript no. 4607751.

The complete per-item experiment records (63,000 case–emotion combinations with prompts, raw responses, post-processed labels, and aggregate evaluation metrics) are published as a companion dataset on Hugging Face:
[gozus19p/emotion-framing-llm-experiments](https://huggingface.co/datasets/gozus19p/emotion-framing-llm-experiments).

## Overview

The study explores how **emotion-associated linguistic framing** affects human–LLM interaction, in both directions of the pipeline:

- **Experiment Alpha** (human → machine) — measures how four emotional prompt styles (**joy**, **apathy**, **anger**, **fear**) influence the performance of five open-weight LLMs across seven SuperGLUE tasks (BoolQ, CB, COPA, ReCoRD, RTE, WiC, WSC).
- **Fine-tuning feasibility study** (machine → human) — develops and evaluates three emotion-conditioned variants (**joy**, **anger**, **fear**) of the Italian-centric [Almawave Velvet-14B](https://huggingface.co/Almawave/Velvet-14B) model via LoRA fine-tuning, and ships the full web application (chat UI + RAG) designed for a prospective user study with students (*Experiment Omega*, presented in the paper as a prospective protocol only — it was not executed).

### Key findings

- Joy- and apathy-framed prompts yield higher model performance, with a cross-task joy–fear gap of up to **5.25 percentage points** (bootstrap 95% CI [4.31, 6.22] pp); item-level exact McNemar tests with Holm correction confirm the joy–fear and apathy–fear contrasts, with small Cohen's *d* effect sizes.
- Lightweight **LoRA fine-tuning** on emotion-annotated (MELD) and synthetic dialogue data induces measurable emotional-tone shifts in Velvet-14B, up to **+19%** joy-related outputs for the joy variant.
- Despite this sensitivity, overall performance remains robust: no catastrophic degradation under any emotional condition.

## Repository structure

```
.
├── alpha/                          # Experiment Alpha: emotion-framed prompting on SuperGLUE
│   ├── logic/
│   │   ├── main.py                 # Typer CLI: inference & evaluation for all models/tasks
│   │   ├── database/               # MongoDB persistence (Case, Evaluation collections)
│   │   ├── experiment/             # Per-task prompt construction & Ollama inference
│   │   └── evaluation/             # Per-task SuperGLUE metrics
│   ├── notebooks/
│   │   ├── sampling/               # SuperGLUE subsampling (≤500 items per task)
│   │   ├── statistics/             # Cohen's d, Cochran's Q, McNemar + Holm, bootstrap
│   │   ├── emotion-perception/     # Emotion-perception validation of the prompt templates
│   │   └── analysis/               # Aggregated analyses
│   └── script/                     # MongoDB helper scripts (mongo.sh, dump.sh)
├── mongodb-collections/             # BSON dump of the results database (Case, Evaluation)
├── omega/                          # Feasibility study & prospective user experiment
│   ├── training/
│   │   ├── training.py             # LoRA fine-tuning of Velvet-14B (PEFT)
│   │   ├── raw_test.py             # Ad-hoc generation testing on the variants
│   │   ├── emotions/               # Synthetic emotion-conditioned datasets (.jsonl)
│   │   └── logs/                   # Training logs for the joy/anger/fear variants
│   ├── logic/
│   │   ├── backend/                # FastAPI service + RAG (Milvus vector database)
│   │   ├── ui/                     # Streamlit chat application
│   │   └── vector-database/        # Milvus docker-compose + DB initialization
│   └── notebooks/                  # MELD data prep, retrieval thresholds, sentiment
│                                   # benchmark, training plots, quantization
├── figures/                        # Figures produced for the paper
├── thesis/                         # Materials of the original thesis project (Italian)
├── requirements-experiment.txt     # Alpha experiment dependencies
├── requirements-training.txt      # Fine-tuning dependencies
├── requirements-backend.txt        # Web backend dependencies
└── requirements-ui.txt             # Web UI dependencies
```

## Prerequisites

- **Python 3.10** (see `.python-version`)
- [Ollama](https://ollama.com) for local LLM execution (Experiment Alpha)
- **Docker** for MongoDB and Milvus
- An Apple Silicon Mac or CUDA GPU for the Velvet-14B fine-tuning (the paper used a MacBook Pro M2 Max, 64 GB, via Metal Performance Shaders)

## Reproducing Experiment Alpha

1. **Install dependencies**:

   ```bash
   pip install -r requirements-experiment.txt
   ```

2. **Pull the models with Ollama** (4-bit quantized, temperature fixed at 0):

   | Model    | Ollama tag                     |
   |----------|--------------------------------|
   | Mistral 7B Instruct | `mistral:7b-instruct-q4_0`   |
   | Llama 3.1 8B Instruct | `llama3.1:8b-instruct-q4_0` |
   | Qwen 2.5 7B Instruct | `qwen2.5:7b-instruct-q4_0`  |
   | Gemma 2 9B Instruct | `gemma2:9b-instruct-q4_0`   |
   | Llama 3.2 3B Instruct | `llama3.2:3b-instruct-q4_0` |

3. **Prepare the data**: download the SuperGLUE datasets (JSONL) and subsample them with `alpha/notebooks/sampling/notebook.ipynb`, which produces `alpha/datasets/<TASK>/sampled.jsonl` (up to 500 items per task). The sampling step is parameterized with an explicit seed in the released code; the original run is fully documented by the per-item records released on Hugging Face.

4. **Start MongoDB**:

   ```bash
   cd alpha
   ./script/mongo.sh
   ```

5. **Run inference and evaluation** (from the `alpha/` directory):

   ```bash
   python -m logic.main main --mode inference
   python -m logic.main main --mode evaluation
   ```

   The model list is defined in `alpha/logic/main.py`; each run stores the per-item interactions (`Case`) and aggregate metrics (`Evaluation`) in the local MongoDB instance. A snapshot of the database used in the paper is provided in `mongodb-collections/`.

6. **Statistical analysis**: use `alpha/notebooks/statistics/notebook.ipynb` for Cohen's *d* effect sizes, Cochran's Q, McNemar tests with Holm correction, and the bootstrap analysis; `alpha/notebooks/emotion-perception/` validates how the prompt templates are perceived.

## Reproducing the fine-tuning feasibility study

```bash
pip install -r requirements-training.txt
cd omega/training
python training.py
```

`training.py` fine-tunes [Almawave/Velvet-14B](https://huggingface.co/Almawave/Velvet-14B) with LoRA (PEFT, targeting `q_proj`/`v_proj`) on the emotion-specific synthetic datasets in `omega/training/emotions/`. Set the desired variant in the `__main__` block (`train(emotion="joy" | "anger" | "fear")`). Checkpoints are written to `./Velvet-14B-<emotion>`; the training logs used in the paper are in `omega/training/logs/`. The training data derives from the emotion-annotated [MELD](https://github.com/declare-lab/MELD) corpus, augmented with synthetic dialogues (see `omega/notebooks/fine-tuning-data-preparation/`); the train/test split uses fixed seed 42.

## Running the web application

The prototype application used for the prospective user experiment consists of a FastAPI backend with RAG retrieval (Milvus) and a Streamlit chat UI.

```bash
# 1. Start Milvus
docker compose -f omega/logic/vector-database/docker-compose.yml up -d
cd omega/logic/vector-database
python db_init.py

# 2. Backend (FastAPI on port 8000)
pip install -r requirements-backend.txt
cd omega/logic/backend
python app.py

# 3. UI (Streamlit)
pip install -r requirements-ui.txt
streamlit run omega/logic/ui/app.py
```

## Data availability

- **Per-item experiment records**: [gozus19p/emotion-framing-llm-experiments](https://huggingface.co/datasets/gozus19p/emotion-framing-llm-experiments)
- **SuperGLUE**: available from the [official benchmark page](https://super.gluebenchmark.com)
- **MELD**: available from [declare-lab/MELD](https://github.com/declare-lab/MELD)
- **Velvet-14B**: available from [Almawave](https://huggingface.co/Almawave/Velvet-14B) on Hugging Face

## Citation

If you use this code or the companion dataset, please cite the paper:

```bibtex
@article{gozzi2026quantifying,
  title   = {Quantifying Emotional Influence in Human--Large Language Model Interaction: Evidence from an Experimental Framework},
  author  = {Gozzi, Manuel and Fallucchi, Francesca},
  journal = {Electronics},
  year    = {2026},
  note    = {Manuscript no. 4607751, under review}
}
```

## License

All rights reserved. The code and materials in this repository are made available to support the inspection and replication of the experiments described in the paper. If you use them, please cite the work.
