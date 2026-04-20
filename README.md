# Diabetes Regression — PyTorch
A neural network regression model built with PyTorch to predict diabetes disease progression using the scikit-learn Diabetes dataset.


# About
This project trains a fully connected neural network to predict the quantitative measure of diabetes disease progression one year after baseline, based on 10 medical features.


# Dataset
sklearn.datasets.load_diabetes

442 samples, 10 features
Features: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6 (blood serum measurements)
Target: continuous value representing disease progression (regression)

# Pipeline
Load dataset and split into train / valid / test sets
Normalize features with StandardScaler (fit on train only)
Convert to PyTorch tensors and create DataLoaders (batch_size=32)
Train model for 20 epochs — tracking loss, train MAE, and valid MAE per epoch
Evaluate final MAE on the held-out test set

# Results
plot_history() plots train vs. valid MAE across epochs
Train MAE ≈ Valid MAE → no overfitting
Final Test MAE is computed and printed separately
