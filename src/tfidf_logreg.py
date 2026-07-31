import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

train_processed = pd.read_csv("Data/processed/train_processed.csv")
test_processed = pd.read_csv("Data/processed/test_processed.csv")

text_replacements = {
    "prompt_text": "[MISSING_PROMPT]",
    "response_a_text": "[MISSING_RESPONSE]",
    "response_b_text": "[MISSING_RESPONSE]",
}

for column, replacement in text_replacements.items():
    train_processed[column] = (
        train_processed[column]
        .fillna(replacement)
        .replace(r"^\s*$", replacement, regex=True)
        .astype(str)
    )


#split train/validation set
train_df, valid_df = train_test_split(train_processed, test_size=0.2, random_state=42, stratify=train_processed['target'])
print("Shape of train set:", train_df.shape)
print("Shape of validation set:", valid_df.shape)

# Calculate the uniform baseline
uniform_predictions = np.full((len(valid_df), 3), 1/3)
uniform_logloss = log_loss(valid_df['target'], uniform_predictions)
print("Première prédiction :", uniform_predictions[0])
print("Uniform baseline log loss:", uniform_logloss)

#tf-idf
prompt_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    lowercase=True,
    sublinear_tf=True
)

response_vectorizer = TfidfVectorizer(
    max_features=15000,  
    ngram_range=(1, 2),
    min_df=2,
    lowercase=True,
    sublinear_tf=True
)

X_train_prompt = prompt_vectorizer.fit_transform(train_df['prompt_text'])
X_valid_prompt = prompt_vectorizer.transform(valid_df['prompt_text'])

all_train_responses =pd.concat([train_df['response_a_text'], train_df['response_b_text']])

response_vectorizer.fit(all_train_responses)
X_train_response_a = response_vectorizer.transform(train_df['response_a_text'])
X_train_response_b = response_vectorizer.transform(train_df['response_b_text'])
X_valid_response_a = response_vectorizer.transform(valid_df['response_a_text'])
X_valid_response_b = response_vectorizer.transform(valid_df['response_b_text'])

X_train_tfidf = hstack([X_train_prompt, X_train_response_a, X_train_response_b])
X_valid_tfidf = hstack([X_valid_prompt, X_valid_response_a, X_valid_response_b])
y_train = train_df['target'].to_numpy()
y_valid = valid_df['target'].to_numpy()

print("X_train :", X_train_tfidf.shape)
print("X_valid :", X_valid_tfidf.shape)

print("Vocabulaire prompt :", len(prompt_vectorizer.vocabulary_))
print("Vocabulaire réponses :", len(response_vectorizer.vocabulary_))

#Training the logistic regression model
log_reg = LogisticRegression(random_state=42,C=0.10)
log_reg.fit(X_train_tfidf, y_train)
valid_probabilities = log_reg.predict_proba(X_valid_tfidf)
print("Première prédiction :", valid_probabilities[0])
print("Validation log loss:", log_loss(y_valid, valid_probabilities))

tfidf_log_loss = log_loss(y_valid, valid_probabilities)
valid_predictions = log_reg.predict(X_valid_tfidf)
valid_accuracy = accuracy_score(y_valid, valid_predictions)

print("Baseline log loss:", uniform_logloss)
print("TF-IDF + Logistic Regression log loss:", tfidf_log_loss)
print("TF-IDF + Logistic Regression accuracy:", valid_accuracy)
print("Amélioration du log loss :", uniform_logloss - tfidf_log_loss)

#Training on the full training set and predicting on the test set
final_prompt_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    lowercase=True,
    sublinear_tf=True
)

final_response_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    min_df=2,
    lowercase=True,
    sublinear_tf=True
)

X_full_prompt = final_prompt_vectorizer.fit_transform(train_processed["prompt_text"])

X_test_prompt = final_prompt_vectorizer.transform(test_processed["prompt_text"])
all_full_responses = pd.concat([train_processed["response_a_text"],train_processed["response_b_text"]])

final_response_vectorizer.fit(all_full_responses)
X_full_a = final_response_vectorizer.transform(train_processed["response_a_text"])

X_full_b = final_response_vectorizer.transform(train_processed["response_b_text"])

X_test_a = final_response_vectorizer.transform(test_processed["response_a_text"])

X_test_b = final_response_vectorizer.transform(test_processed["response_b_text"])
X_full_tfidf = hstack([X_full_prompt,X_full_a,X_full_b])
X_test_tfidf = hstack([X_test_prompt,X_test_a,X_test_b])
y_full = train_processed["target"].to_numpy()

print("Train complet :", X_full_tfidf.shape)
print("Test :", X_test_tfidf.shape)
print("Cible :", y_full.shape)
final_model = LogisticRegression(C=0.1,random_state=42)

final_model.fit(X_full_tfidf, y_full)

print("Classes :", final_model.classes_)
print("Nombre d'itérations :", final_model.n_iter_)

test_probabilities = final_model.predict_proba(X_test_tfidf)

print("Forme :", test_probabilities.shape)
print("Classes :", final_model.classes_)
print("\nProbabilités :")
print(test_probabilities)

print("Sommes des probabilités :",test_probabilities.sum(axis=1))

print("Toutes les sommes valent 1 :",np.allclose(test_probabilities.sum(axis=1),1))