import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. load dataset and split
dataset = load_iris()
features = dataset.data
targets = dataset.target
classNames = dataset.target_names

trainX, testX, trainY, testY = train_test_split(
    features,
    targets,
    test_size=0.2,
    random_state=42
)

# 2. train baseline model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(trainX, trainY)

originalPredictions = model.predict(testX)
originalAccuracy = accuracy_score(testY, originalPredictions)
print("Original Model Accuracy:", round(originalAccuracy, 4))

# 3. serialize using pickle
pickleFileName = "rfModelPickle.pkl"
with open(pickleFileName, "wb") as fileHandle:
    pickle.dump(model, fileHandle)
print("Saved model via pickle to:", pickleFileName)

# deserialize using pickle
with open(pickleFileName, "rb") as fileHandle:
    loadedPickleModel = pickle.load(fileHandle)

picklePredictions = loadedPickleModel.predict(testX)
pickleAccuracy = accuracy_score(testY, picklePredictions)
print("Pickle Reloaded Accuracy:", round(pickleAccuracy, 4))
print("Pickle Match Verification:", np.array_equal(originalPredictions, picklePredictions))

# 4. serialize using joblib (optimized for scikit-learn and large numpy arrays)
joblibFileName = "rfModelJoblib.joblib"
joblib.dump(model, joblibFileName)
print("Saved model via joblib to:", joblibFileName)

# deserialize using joblib
loadedJoblibModel = joblib.load(joblibFileName)
joblibPredictions = loadedJoblibModel.predict(testX)
joblibAccuracy = accuracy_score(testY, joblibPredictions)
print("Joblib Reloaded Accuracy:", round(joblibAccuracy, 4))
print("Joblib Match Verification:", np.array_equal(originalPredictions, joblibPredictions))

# 5. create and save confusion matrix plot as deliverable image evidence
confMatrix = confusion_matrix(testY, joblibPredictions)
displayMatrix = ConfusionMatrixDisplay(confusion_matrix=confMatrix, display_labels=classNames)

fig, ax = plt.subplots(figsize=(6, 5))
displayMatrix.plot(ax=ax, cmap="Blues")
plt.title("Reloaded Joblib Model Predictions")
plt.tight_layout()
plt.savefig("modelEvaluationMatrix.png")
plt.show()