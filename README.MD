# LLM Classification Fine-Tuning

This project is based on the Kaggle competition **LLM Classification Fine-Tuning**.

The objective is to predict which response is preferred for a given prompt:

- `0`: Model A wins
- `1`: Model B wins
- `2`: Tie

## Approach

The current baseline uses:

- JSON conversation preprocessing
- missing-value handling
- TF-IDF vectorization
- logistic regression
- multiclass log loss for evaluation

The prompt, response A and response B are vectorized separately before being combined into one feature matrix.

## Next Version

In the next version, I will implement a Transformer-based classifier and compare its performance with the current TF-IDF and logistic regression baseline.

## Dataset

The dataset is provided through the Kaggle competition.

Due to the competition rules, the dataset is not included or redistributed in this repository.

To run the project:

1. Join the competition and accept its rules.
2. Download `train.csv`, `test.csv` and `sample_submission.csv`.
3. Place them inside `Data/raw/`.

## Run the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

or use the makefile :

```bash
make install-requirements
```
