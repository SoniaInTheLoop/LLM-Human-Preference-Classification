import pandas as pd
import json
import numpy as np
from pathlib import Path



train = pd.read_csv("Data/raw/train.csv")
test = pd.read_csv("Data/raw/test.csv")
submission = pd.read_csv("Data/raw/sample_submission.csv")


# Data analysis
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Submission shape:", submission.shape)

print("\nTrain columns:", train.columns.to_list())
print("Test columns:", test.columns.to_list())
print("Submission columns:", submission.columns.to_list())

print("\nTrain preview:")
print(train.head())

print("\nTest preview:")
print(test.head())

print("\nSubmission preview:")
print(submission.head())


# Inspect the first training example: prompt, response A and response B
prompt_example = train.loc[0, "prompt"]
response_a_example = train.loc[0, "response_a"]
response_b_example = train.loc[0, "response_b"]

print("\nPrompt example:", prompt_example)
print("\nResponse A example:", response_a_example)
print("\nResponse B example:", response_b_example)

print("\nPrompt length:", len(prompt_example))
print("Prompt type:", type(prompt_example))

print("\nResponse A length:", len(response_a_example))
print("Response A type:", type(response_a_example))

print("\nResponse B length:", len(response_b_example))
print("Response B type:", type(response_b_example))


# Convert a JSON string into a Python list
def convert_json_to_list(value: str) -> list:
    return json.loads(value)


# Convert the first example
prompt_list = convert_json_to_list(prompt_example)
response_a_list = convert_json_to_list(response_a_example)
response_b_list = convert_json_to_list(response_b_example)

print("\nPrompt list:", prompt_list)
print("Prompt list type:", type(prompt_list))
print("Prompt list length:", len(prompt_list))

print("\nResponse A list:", response_a_list)
print("Response A list type:", type(response_a_list))
print("Response A list length:", len(response_a_list))

print("\nResponse B list:", response_b_list)
print("Response B list type:", type(response_b_list))
print("Response B list length:", len(response_b_list))


# Inspect the models and winner of the first example
print("\nModel A:", train.loc[0, "model_a"])
print("Model B:", train.loc[0, "model_b"])
print("Winner model A:", train.loc[0, "winner_model_a"])
print("Winner model B:", train.loc[0, "winner_model_b"])
print("Winner tie:", train.loc[0, "winner_tie"])


# Display each conversation turn from the first example
'''
for i in range(len(prompt_list)):
    print("-" * 50)
    print("Turn:", i + 1)
    print("\nPrompt:", prompt_list[i])
    print("\nResponse A:", response_a_list[i])
    print("\nResponse B:", response_b_list[i])
'''

# Convert the entire prompt, response_a and response_b columns
train["prompt_list"] = train["prompt"].apply(convert_json_to_list)
train["response_a_list"] = train["response_a"].apply(convert_json_to_list)
train["response_b_list"] = train["response_b"].apply(convert_json_to_list)


# Verify the type stored in the new columns
print("\nTypes after conversion:")
print(type(train.loc[0, "prompt_list"]))
print(type(train.loc[0, "response_a_list"]))
print(type(train.loc[0, "response_b_list"]))

# Check that each conversation contains the same number of prompts and responses
train["n_prompts"]=train["prompt_list"].apply(len)
train["n_response_a"]=train["response_a_list"].apply(len)
train["n_response_b"]=train["response_b_list"].apply(len)

aligned = (train["n_prompts"] == train["n_response_a"]) & (train["n_prompts"] == train["n_response_b"])
print("\nNumber of aligned conversations:", aligned.sum())
print("Number of misaligned conversations:", (~aligned).sum())

# Verify that each training sample has exactly one winner label
winner_labels_sum = train[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1)
invalid_count = 0
for i in range(len(train)):
    if winner_labels_sum.iloc[i] != 1:
        print(f"Conversation {i} has an invalid winner label configuration.")
        invalid_count += 1

print("\nNumber of conversations with invalid winner label configuration:", invalid_count)

# Analyze the number and proportion of wins for each outcome
winner_columns = ["winner_model_a", "winner_model_b", "winner_tie"]

print("\nWinner counts:")
print(train[winner_columns].sum())

print("\nWinner proportions:")
print(train[winner_columns].mean())

# Encode the winner labels into a single target column:
train["target"] = np.select([
    train["winner_model_a"]==1,
    train["winner_model_b"]==1,
    train["winner_tie"]==1],
    [0, 1, 2],
    default= -1
)
print(train["target"].value_counts())

# Analyze the distribution and summary statistics of the number of prompts per conversation
train["n_prompts"].value_counts()
train["n_prompts"].describe()

# Check for missing values in the dataset
print("Missing values in the train data:", train.isna().sum())
print("Missing values in the test data:", test.isna().sum())

# Check for non-string values before joining response messages for TF-IDF
invalid_a = train["response_a_list"].apply(
    lambda messages: any(not isinstance(message, str)
                          for message in messages))

invalid_b = train["response_b_list"].apply(
    lambda messages: any(not isinstance(message, str)                      
                          for message in messages))

print("Réponses A contenant un élément non-string :", invalid_a.sum())
print("Réponses B contenant un élément non-string :", invalid_b.sum())

train["response_a_text"] = train["response_a_list"].apply(
    lambda messages: " ".join(message if isinstance(message, str) 
                            else "MISSING RESPONSE" for message in messages))

train["response_b_text"] = train["response_b_list"].apply(
    lambda messages: " ".join(message if isinstance(message, str)
                            else "MISSING RESPONSE" for message in messages))

train["prompt_text"] = train["prompt_list"].apply(
    lambda messages : " ".join(message if isinstance(message, str)
                            else "MISSING_PROMPT"
                            for message in messages))

# Replace empty texts before saving the processed dataset
train["prompt_text"] = train["prompt_text"].replace(
    r"^\s*$",
    "[MISSING_PROMPT]",
    regex=True,
)

train["response_a_text"] = train["response_a_text"].replace(
    r"^\s*$",
    "[MISSING_RESPONSE]",
    regex=True,
)

train["response_b_text"] = train["response_b_text"].replace(
    r"^\s*$",
    "[MISSING_RESPONSE]",
    regex=True,
)

# Analyze whether the longer or shorter response wins
length_analysis = train.loc[~invalid_a & ~invalid_b].copy()
print("Nombre de conversations analysées: ", len(length_analysis))

length_analysis["n_words_a"] = (
    length_analysis["response_a_text"].str.split().str.len()
)

length_analysis["n_words_b"] = (
    length_analysis["response_b_text"].str.split().str.len()
)

length_analysis["word_difference"] = (length_analysis["n_words_a"]- length_analysis["n_words_b"])

length_analysis["longer_response"] = np.select(
[
    length_analysis["n_words_a"] > length_analysis["n_words_b"],
    length_analysis["n_words_a"] < length_analysis["n_words_b"]
],
["A_longer","B_longer"],default="same_length")

unequal_length = (length_analysis["longer_response"] != "same_length")

longer_wins = (
    ((length_analysis["longer_response"] == "A_longer")& (length_analysis["target"] == 0))|
    ((length_analysis["longer_response"] == "B_longer")& (length_analysis["target"] == 1))
)

shorter_wins = (
    ((length_analysis["longer_response"] == "A_longer") & (length_analysis["target"] == 1))|
    ((length_analysis["longer_response"] == "B_longer") & (length_analysis["target"] == 0))
)

ties_unequal = (unequal_length & (length_analysis["target"] == 2))

print("La réponse la plus longue gagne :",
      round(longer_wins.sum() / unequal_length.sum() * 100, 2),"%")

print("La réponse la plus courte gagne :",
    round(shorter_wins.sum() / unequal_length.sum() * 100, 2),"%")

print("Égalité malgré des longueurs différentes :",
    round(ties_unequal.sum() / unequal_length.sum() * 100, 2),"%")

#Model imput 
# Create the model input text
train["input_model"] = (
    "PROMPT: " + train["prompt_text"]
    + "\nRESPONSE A: " + train["response_a_text"]
    + "\nRESPONSE B: " + train["response_b_text"])

processed_dir = Path("Data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(
    processed_dir / "train_processed.csv",
    index=False,)

print("\nProcessed training data saved successfully.")

#test preprocessing
test["prompt_list"] = test["prompt"].apply(convert_json_to_list)
test["response_a_list"] = test["response_a"].apply(convert_json_to_list)
test["response_b_list"] = test["response_b"].apply(convert_json_to_list)

test["prompt_text"] = test["prompt_list"].apply(
    lambda messages: " ".join(
        message if isinstance(message, str)
        else "MISSING_PROMPT"
        for message in messages
    )
)

test["response_a_text"] = test["response_a_list"].apply(
    lambda messages: " ".join(
        message if isinstance(message, str)
        else "MISSING_RESPONSE"
        for message in messages
    )
)

test["response_b_text"] = test["response_b_list"].apply(
    lambda messages: " ".join(
        message if isinstance(message, str)
        else "MISSING_RESPONSE"
        for message in messages
    )
)

test.to_csv(
    processed_dir / "test_processed.csv",
    index=False,
)

print("\nProcessed test data saved successfully.")