import torch
from pykan.kan.MultKAN import KAN
from scripts.preprocessing.data_loader import get_train_test_splits
from scripts.preprocessing.preprocessor import apply_minmax_scaling
import numpy as np
import pandas as pd

X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.2)
X_train, y_train, train_scales = apply_minmax_scaling(X_train, y_train)
X_test, y_test, test_scales = apply_minmax_scaling(X_test, y_test)

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Convert your data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float64)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float64)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float64)

# Create a dataset dictionary with both training and testing data
dataset = {
    'train_input': X_train_tensor,
    'train_output': y_train_tensor,
    'test_input': X_test_tensor,
    'test_output': y_test_tensor
}

# Create the KAN model
model = KAN(width=[19, 64, 32, 3], device=device)

# Train the model
model.fit(dataset, steps=1000, lamb=0.001)

# Make predictions
y_pred = model.predict(X_test_tensor)

# Convert predictions back to numpy for evaluation
y_pred_np = y_pred.cpu().numpy()

from sklearn.metrics import mean_squared_error, r2_score

# Assuming your targets are continuous variables
mse = mean_squared_error(y_test, y_pred_np)
r2 = r2_score(y_test, y_pred_np)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

model.plot()