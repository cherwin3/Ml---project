import zipfile, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with zipfile.ZipFile("Spam_msg.zip", "r") as zip_ref:
    zip_ref.extractall("spam_data")

csv_file = [f for f in os.listdir("spam_data") if f.endswith(".csv") or f.endswith(".tsv")][0]

if csv_file.endswith(".tsv"):
    df = pd.read_csv(f"spam_data/{csv_file}", sep="\t", header=None)
    df.columns = ["label", "message"]
else:
    df = pd.read_csv(f"spam_data/{csv_file}")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
