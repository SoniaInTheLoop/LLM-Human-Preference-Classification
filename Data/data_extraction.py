import pandas as pd 
from pathlib import Path

DATA_DIR=Path(r"C:\Users\sonia\Downloads\llm-classification-finetuning (1)")

train = pd.read_csv(DATA_DIR/ "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

PROJECT_DATA_DIR = Path("Data/raw")
PROJECT_DATA_DIR.mkdir(parents= True, exist_ok= True)

train.to_csv(PROJECT_DATA_DIR / "train.csv", index=False)
test.to_csv(PROJECT_DATA_DIR / "test.csv", index=False)
sample_submission.to_csv(PROJECT_DATA_DIR / "sample_submission.csv", index=False)   

print("Data extraction completed successfully. Files saved in 'Data/raw' directory")



