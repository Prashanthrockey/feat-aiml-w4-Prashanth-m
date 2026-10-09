import os
import sys
import logging
import warnings

# mute logging and standard warnings
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# mute third party import prints on stderr
sysStderrOriginal = sys.stderr
sys.stderr = open(os.devnull, "w")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# restore stderr
sys.stderr = sysStderrOriginal

# generate synthetic demographic data with clear systemic wage bias
np.random.seed(42)
sampleCount = 1000

gender = np.random.binomial(1, 0.60, sampleCount)
educationYears = np.random.normal(12, 3, sampleCount).clip(6, 18)
hoursPerWeek = np.random.normal(40, 10, sampleCount).clip(15, 70)
age = np.random.normal(38, 12, sampleCount).clip(18, 75)

# explicit advantage given to privileged group (gender 1)
incomeScores = 0.05 * age + 0.1 * educationYears + 0.02 * hoursPerWeek + 1.8 * gender - 4.2
incomeProbabilities = 1 / (1 + np.exp(-incomeScores))
incomeLabels = (incomeProbabilities > 0.5).astype(int)

df = pd.DataFrame({
    "age": age,
    "educationYears": educationYears,
    "hoursPerWeek": hoursPerWeek,
    "gender": gender,
    "income": incomeLabels
})

privilegedGroups = [{"gender": 1}]
unprivilegedGroups = [{"gender": 0}]

aifDataset = BinaryLabelDataset(
    df=df,
    label_names=["income"],
    protected_attribute_names=["gender"],
    favorable_label=1,
    unfavorable_label=0
)

# 1. baseline disparate impact
baselineMetric = BinaryLabelDatasetMetric(
    aifDataset,
    unprivileged_groups=unprivilegedGroups,
    privileged_groups=privilegedGroups
)
baselineDisparateImpact = baselineMetric.disparate_impact()
print("Task 1 - Baseline Disparate Impact:", round(baselineDisparateImpact, 4))
print("Task 1 - Violates 80 Percent Rule:", baselineDisparateImpact < 0.8)

# 2. train model and save shap summary plot
featureNames = ["age", "educationYears", "hoursPerWeek", "gender"]
trainX = df[featureNames]
trainY = df["income"]

scaler = StandardScaler()
trainXScaled = scaler.fit_transform(trainX)

model = LogisticRegression(random_state=42)
model.fit(trainXScaled, trainY)

# set max_samples to match sample size to suppress shap sampling notice
masker = shap.maskers.Independent(trainXScaled, max_samples=1000)
explainer = shap.LinearExplainer(model, masker)
shapValues = explainer(trainXScaled)

plt.figure(figsize=(8, 4))
shap.plots.beeswarm(shapValues, show=False)
plt.title("Task 2 - SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shapSummaryPlot.png")
plt.close()
print("Task 2 - SHAP summary plot saved to shapSummaryPlot.png")

# 3. apply reweighing mitigation
reweighAlgorithm = Reweighing(
    unprivileged_groups=unprivilegedGroups,
    privileged_groups=privilegedGroups
)
transformedDataset = reweighAlgorithm.fit_transform(aifDataset)

mitigatedMetric = BinaryLabelDatasetMetric(
    transformedDataset,
    unprivileged_groups=unprivilegedGroups,
    privileged_groups=privilegedGroups
)
mitigatedDisparateImpact = mitigatedMetric.disparate_impact()
print("Task 3 - Mitigated Disparate Impact:", round(mitigatedDisparateImpact, 4))
print("Task 3 - Fairness Improvement Delta:", round(mitigatedDisparateImpact - baselineDisparateImpact, 4))