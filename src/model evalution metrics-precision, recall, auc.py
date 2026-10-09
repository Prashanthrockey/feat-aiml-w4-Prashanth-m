import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# create sample dataset with slight imbalance
features, labels = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=6,
    weights=[0.8, 0.2],
    random_state=42
)

# train test split
trainX, testX, trainY, testY = train_test_split(
    features,
    labels,
    test_size=0.25,
    stratify=labels,
    random_state=42
)

# initialize and train model
model = LogisticRegression(max_iter=300)
model.fit(trainX, trainY)

# predictions
predictions = model.predict(testX)
probabilities = model.predict_proba(testX)[:, 1]

# calculate individual metrics
precisionVal = precision_score(testY, predictions, zero_division=0)
recallVal = recall_score(testY, predictions)
f1Val = f1_score(testY, predictions)
aucVal = roc_auc_score(testY, probabilities)

print("Evaluation Results on Test Set:")
print("Precision:", round(precisionVal, 4))
print("Recall:", round(recallVal, 4))
print("F1 Score:", round(f1Val, 4))
print("ROC AUC:", round(aucVal, 4))

# stratified k-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvScores = cross_val_score(model, features, labels, cv=kfold, scoring="roc_auc")
print("Stratified CV Mean ROC AUC:", round(cvScores.mean(), 4))

# roc curve plot
fpr, tpr, thresholds = roc_curve(testY, probabilities)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="Model ROC")
plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("rocCurvePlot.png")
plt.show()

# learning curve to check overfitting vs underfitting
trainSizes, trainScores, valScores = learning_curve(
    model,
    features,
    labels,
    cv=5,
    scoring="f1",
    train_sizes=np.linspace(0.1, 1.0, 5),
    random_state=42
)

plt.figure(figsize=(6, 4))
plt.plot(trainSizes, np.mean(trainScores, axis=1), label="Train F1")
plt.plot(trainSizes, np.mean(valScores, axis=1), label="Validation F1")
plt.xlabel("Training Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve Diagnostic")
plt.legend()
plt.tight_layout()
plt.savefig("learningCurvePlot.png")
plt.show()
