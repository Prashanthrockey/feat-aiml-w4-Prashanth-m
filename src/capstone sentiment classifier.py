import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# 1. binary sentiment classification dataset
texts = [
    "great product loved it completely",
    "terrible experience will never buy again",
    "awesome quality very happy",
    "poor quality broke on first day",
    "highly recommended works perfectly",
    "worst customer support useless item",
    "five stars completely satisfied",
    "defective and totally disappointing",
    "excellent build and superb speed",
    "waste of money very unhappy",
    "smooth performance super easy setup",
    "horrible packaging came broken",
    "fantastic tool saves so much time",
    "do not recommend bad material",
    "love this device great value"
] * 40

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 40

trainTexts, testTexts, trainY, testY = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# 2. task 1: train logistic regression pipeline and print report
logisticPipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=500)),
    ("clf", LogisticRegression(random_state=42))
])

logisticPipeline.fit(trainTexts, trainY)
logisticPredictions = logisticPipeline.predict(testTexts)
logisticProbabilities = logisticPipeline.predict_proba(testTexts)[:, 1]

print("=== Logistic Regression Classification Report ===")
print(classification_report(testY, logisticPredictions, target_names=["Negative", "Positive"]))

# 3. task 3: train random forest pipeline and compare
rfPipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=500)),
    ("clf", RandomForestClassifier(n_estimators=50, random_state=42))
])

rfPipeline.fit(trainTexts, trainY)
rfPredictions = rfPipeline.predict(testTexts)
rfProbabilities = rfPipeline.predict_proba(testTexts)[:, 1]

# comparison metrics table
print("=== Model Performance Comparison ===")
print("Metric         | Logistic Reg | Random Forest")
print("Accuracy       | " + str(round(accuracy_score(testY, logisticPredictions), 4)).ljust(12) + " | " + str(round(accuracy_score(testY, rfPredictions), 4)))
print