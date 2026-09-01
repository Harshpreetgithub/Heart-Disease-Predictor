import pandas as pd
import numpy as np
import pickle, json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("heart.csv")

# ---- Encoding (same as report Chapter 4 / 8) ----
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])          # F=0, M=1
le_ex = LabelEncoder()
df["ExerciseAngina"] = le_ex.fit_transform(df["ExerciseAngina"])  # N=0, Y=1

df = pd.get_dummies(df, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=True)

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]
feature_columns = list(X.columns)   # exact column order the API must replicate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Classification": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cv = cross_val_score(model, scaler.transform(X), y, cv=10).mean()
    results[name] = dict(test_acc=round(acc*100,2), precision=round(prec,2),
                          recall=round(rec,2), f1=round(f1,2), cv10=round(cv*100,2))
    print(f"{name:32s} acc={acc*100:.2f}%  prec={prec:.2f}  rec={rec:.2f}  f1={f1:.2f}  cv10={cv*100:.2f}%")

# Final deployed model = Random Forest (best + most interpretable, per report Chapter 6/7)
final_model = models["Random Forest"]

with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump({"Sex": le_sex, "ExerciseAngina": le_ex}, f)
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFeature columns (order matters):")
print(feature_columns)
