# Financial News Sentiment Analysis  
_Natural Language Processing – Technische Universität Berlin (Master’s in Computer Science)_

## Overview
This project was developed as part of the Natural Language Processing course during the Master’s in Computer Science program at Technische Universität Berlin. The objective is to classify the sentiment of financial news headlines as **positive**, **neutral**, or **negative** using classical machine learning and neural network methods.

---
## Dataset

The dataset used is the **Financial Phrase Bank v1.0**, created by researchers at Aalto University. It contains 4,840 financial news sentences annotated with sentiment labels: *positive*, *neutral*, or *negative*. This project uses the `Sentences_50Agree.txt` file, which includes sentences with at least 50% annotator agreement.
For full dataset documentation and license details, see:
- [`data/README`](data/README) – Dataset description and annotation process  
- [`data/LICENSE`](data/LICENSE) – Creative Commons BY-NC-SA 3.0 License

---
## Objectives
- Explore and visualize dataset insights
- Pre-process raw financial text data
- Build and evaluate:
  - Naïve Bayes classifier
  - Feed-forward neural network
  - Binary sentiment classifier (positive vs. negative)
- Perform error analysis and sentence-level similarity
- Optional: Build a BERT-based RNN for advanced classification
---
## Project Structure
```
financial-news-sentiment-analysis/
│
├── data/                       # Dataset files
├── notebooks/                  # Jupyter notebooks
├── src/                        # Scripts for models, preprocessing, evaluation
├── reports/                    # Results and Final report
├── requirements.txt            # Python package dependencies
└── README.md                   # Project overview
  ```
---
## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aimanalhazmi/financial-news-sentiment-nlp.git
cd financial-news-sentiment-nlp
```

### 2. Set up the environment using `make`
```bash
make
```

This will:
- Create a virtual environment in `.venv/`
- Register a Jupyter kernel as **'financial-news-sentiment-nlp'**
- Install all required packages from `requirements.txt`

### 3. Activate the environment
```bash
source .venv/bin/activate
```

## ⚙️ Makefile Commands

| Command             | Description                                                |
|---------------------|------------------------------------------------------------|
| `make install`      | Set up virtual environment and install dependencies        |
| `make activate`     | Print the command to activate the environment              |
| `make jupyter-kernel` | Register Jupyter kernel as `financial-news-sentiment-nlp`          |
| `make remove-kernel`  | Unregister existing kernel (if needed)                  |
| `make clean`        | Delete the virtual environment folder                      |


---

## Example Usage
```bash
python3 train_evaluate.py
```
---
## Hyperparameter Tuning with wandb Sweeps
This project supports automated hyperparameter tuning using wandb sweeps.

### Sweep Configuration
The sweep uses grid search to tune the following parameters:
- lr: Learning rate 
- weight_decay: L2 regularization 
- label_smoothing: Label smoothing factor 
- hidden_dim: Size of hidden layer 
- vectorizer: Vectorization strategy 
- epochs: Fixed at 30 
- metric: Optimized for best_model/f1_score

[Sweep config (YAML format)](sweep.yaml)

### How to Run the Sweep

#### 1. Log in to wandb (one-time setup)

```bash
wandb login
```
#### 2. Initialize the sweep:
```bash
wandb sweep sweep.yaml
```
This command will output a sweep ID, e.g., username/project/sweepid123.
#### 3. Launch agents to run the sweep:
```bash
wandb agent username/project/sweepid123
```
Each agent will run one combination of parameters until all are explored.
You can then view and compare runs at: https://wandb.ai.