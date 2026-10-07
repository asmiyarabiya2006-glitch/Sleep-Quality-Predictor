import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sleep.csv")

def make_label(score):
    if score <= 6:
        return "Poor"
    elif score == 7:
        return "Average"
    else:
        return "Good"

df["Sleep Label"] = df["Quality of Sleep"].apply(make_label)

features = ["Sleep Duration", "Physical Activity Level", "Stress Level"]

X = df[features]
y = df["Sleep Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", X_train.shape)
print("Testing rows:", X_test.shape)


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

print("\nMODEL ACCURACY:")
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(name, ":", round(score * 100, 2), "%")


from sklearn.metrics import classification_report, confusion_matrix

best_model = models["Random Forest"]
predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))

print("CONFUSION MATRIX:")
labels = ["Poor", "Average", "Good"]
print(confusion_matrix(y_test, predictions, labels=labels))

print("\nFEATURE IMPORTANCE:")
for feature, score in zip(features, best_model.feature_importances_):
    print(feature, ":", round(score, 3)) 


import joblib

joblib.dump(best_model, "model/sleep_model.pkl")
print("\nModel saved to model/sleep_model.pkl")