import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# generate synthetic noisy regression dataset to demonstrate bias variance
features, targets = make_regression(
    n_samples=250,
    n_features=40,
    n_informative=10,
    noise=15.0,
    random_state=42
)

# train test split
trainX, testX, trainY, testY = train_test_split(
    features,
    targets,
    test_size=0.25,
    random_state=42
)

# 1. baseline unregularized model (high variance / overfitting potential)
baseModel = LinearRegression()
baseModel.fit(trainX, trainY)
baseTrainPred = baseModel.predict(trainX)
baseTestPred = baseModel.predict(testX)

print("Baseline Linear Regression (Unregularized):")
print("Train MSE:", round(mean_squared_error(trainY, baseTrainPred), 3))
print("Test MSE:", round(mean_squared_error(testY, baseTestPred), 3))
print("Test R2 Score:", round(r2_score(testY, baseTestPred), 3))

# 2. grid search on ridge regression (L2 regularization)
ridgeParams = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]}
ridgeGrid = GridSearchCV(
    estimator=Ridge(),
    param_grid=ridgeParams,
    cv=5,
    scoring="neg_mean_squared_error"
)
ridgeGrid.fit(trainX, trainY)
bestRidge = ridgeGrid.best_estimator_

print("\nRidge Regression (Grid Search):")
print("Best Alpha:", ridgeGrid.best_params_["alpha"])
ridgeTestPred = bestRidge.predict(testX)
print("Test MSE:", round(mean_squared_error(testY, ridgeTestPred), 3))
print("Test R2 Score:", round(r2_score(testY, ridgeTestPred), 3))

# 3. randomized search on lasso regression (L1 regularization / feature sparsity)
lassoDistributions = {"alpha": np.logspace(-2, 2, 50)}
lassoRandom = RandomizedSearchCV(
    estimator=Lasso(max_iter=5000),
    param_distributions=lassoDistributions,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42
)
lassoRandom.fit(trainX, trainY)
bestLasso = lassoRandom.best_estimator_

print("\nLasso Regression (Randomized Search):")
print("Best Alpha:", round(lassoRandom.best_params_["alpha"], 4))
lassoTestPred = bestLasso.predict(testX)
print("Test MSE:", round(mean_squared_error(testY, lassoTestPred), 3))
print("Test R2 Score:", round(r2_score(testY, lassoTestPred), 3))
print("Non-zero Features Kept by Lasso:", np.sum(bestLasso.coef_ != 0), "out of", trainX.shape[1])

# 4. plot coefficient comparison (visual evidence of regularization shrink vs sparsity)
plt.figure(figsize=(9, 4))
plt.plot(baseModel.coef_, label="Linear (No Reg)", alpha=0.5, color="red")
plt.plot(bestRidge.coef_, label="Ridge L2 (Shrunk)", alpha=0.8, color="blue")
plt.plot(bestLasso.coef_, label="Lasso L1 (Sparse)", alpha=0.8, color="green")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Weight")
plt.title("Regularization Effect on Feature Coefficients")
plt.legend()
plt.tight_layout()
plt.savefig("regularizationComparisonPlot.png")
plt.show()

# 5. plot bias-variance trade-off across alphas for ridge
alphas = np.logspace(-2, 3, 20)
trainErrors = []
testErrors = []

for a in alphas:
    tempModel = Ridge(alpha=a)
    tempModel.fit(trainX, trainY)
    trainErrors.append(mean_squared_error(trainY, tempModel.predict(trainX)))
    testErrors.append(mean_squared_error(testY, tempModel.predict(testX)))

plt.figure(figsize=(7, 4))
plt.plot(alphas, trainErrors, label="Train Error (Bias Trend)", color="orange")
plt.plot(alphas, testErrors, label="Test Error (Variance Control)", color="purple")
plt.xscale("log")
plt.xlabel("Alpha Regularization Strength")
plt.ylabel("Mean Squared Error")
plt.title("Bias-Variance Tradeoff Curve")
plt.legend()
plt.tight_layout()
plt.savefig("biasVarianceCurvePlot.png")
plt.show()