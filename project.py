

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import seaborn as sns
import numpy as np

path = kagglehub.dataset_download("shantanugarg274/sales-dataset")
print("Dataset path:", path)


print("Files in dataset folder:")
print(os.listdir(path))  


df = pd.read_csv(os.path.join(path, "Sales Dataset.csv"))


print("Initial Data Sample:")
print(df.head())
print("Column names:", df.columns.tolist())

df = df.dropna()

df = pd.get_dummies(df, drop_first=True)

median_sales = df['Amount'].median()
df['SalesCategory'] = df['Amount'].apply(lambda x: 'High' if x > median_sales else 'Low')

df = df.drop('Amount', axis=1)

X = df.drop('SalesCategory', axis=1)
y = df['SalesCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, pos_label='High'))
    print("Recall   :", recall_score(y_true, y_pred, pos_label='High'))
    print("F1 Score :", f1_score(y_true, y_pred, pos_label='High'))
    print("\n")

evaluate_model("Decision Tree", y_test, dt_preds)
evaluate_model("K-Nearest Neighbors", y_test, knn_preds)

results = {
    "Model": ["Decision Tree", "KNN"],
    "Accuracy": [accuracy_score(y_test, dt_preds), accuracy_score(y_test, knn_preds)],
    "Precision": [precision_score(y_test, dt_preds, pos_label='High'), precision_score(y_test, knn_preds, pos_label='High')],
    "Recall": [recall_score(y_test, dt_preds, pos_label='High'), recall_score(y_test, knn_preds, pos_label='High')],
    "F1 Score": [f1_score(y_test, dt_preds, pos_label='High'), f1_score(y_test, knn_preds, pos_label='High')],
}

result_df = pd.DataFrame(results)
print("Model Comparison Summary:")
print(result_df)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = range(len(metrics))

dt_values = result_df[result_df["Model"] == "Decision Tree"].iloc[0, 1:].values
knn_values = result_df[result_df["Model"] == "KNN"].iloc[0, 1:].values

bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - bar_width/2 for i in x], dt_values, width=bar_width, label="Decision Tree", color='skyblue')
plt.bar([i + bar_width/2 for i in x], knn_values, width=bar_width, label="KNN", color='lightgreen')

plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Model Comparison: Decision Tree vs KNN")
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=["High", "Low"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["High", "Low"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {title}")
    plt.show()

plot_confusion_matrix(y_test, dt_preds, "Decision Tree")
plot_confusion_matrix(y_test, knn_preds, "K-Nearest Neighbors")

def plot_roc_curve(model, X_test, y_test, title):
    y_test_binary = (y_test == 'High').astype(int)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_binary, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{title} (AUC = {roc_auc:.2f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()
plot_roc_curve(dt_model, X_test, y_test, "Decision Tree")
plot_roc_curve(knn_model, X_test, y_test, "K-Nearest Neighbors")