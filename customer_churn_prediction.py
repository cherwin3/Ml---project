import zipfile, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

with zipfile.ZipFile("Bank.zip", "r") as zip_ref:
    zip_ref.extractall("bank_data")

csv_file = [f for f in os.listdir("bank_data") if f.endswith(".csv")][0]
df = pd.read_csv(f"bank_data/{csv_file}")

possible_targets = ["Churn", "Exited", "churn", "exit", "ExitedCustomer", "target"]
target = None
for col in possible_targets:
    if col in df.columns:
        target = col
        break

drop_cols = ["RowNumber", "CustomerId", "Surname"]
df = df.drop(columns=drop_cols, errors="ignore")

label_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
