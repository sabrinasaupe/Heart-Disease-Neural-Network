import numpy as np
import pandas as pd


df = pd.read_csv('heart.csv')
print(f"=========================================================")
print(f"               DATA INGESTION PROFILE STATE               ")
print(f"=========================================================")
print(f"Raw Matrix Dimensions Detected : {df.shape[0]} rows, {df.shape[1]} columns.")

df = df.dropna()
print(f"Cleaned Matrix Dimensions      : {df.shape[0]} rows, {df.shape[1]} columns.")


X_raw = df.drop(columns=['target']).values


y = df['target'].values.reshape(-1, 1)


feat_min = X_raw.min(axis=0)
feat_max = X_raw.max(axis=0)

print(f"\nCaptured Feature Calibration Bounds:")
print(f"-> Base Matrix Lowest Limits  (feat_min) :\n   {feat_min}")
print(f"-> Base Matrix Highest Limits (feat_max) :\n   {feat_max}")


X_scaled = (X_raw - feat_min) / (feat_max - feat_min)


np.random.seed(42)


indices = np.arange(len(X_scaled))


np.random.shuffle(indices)


split_threshold = int(0.8 * len(indices))


X_train = X_scaled[indices[:split_threshold]]
X_test = X_scaled[indices[split_threshold:]]
y_train = y[indices[:split_threshold]]
y_test = y[indices[split_threshold:]]


print(f"\n=========================================================")
print(f"               MATRIX DATA DIVISION STATUS               ")
print(f"=========================================================")
print(f"Training Input Array (X_train) Shape : {X_train.shape} -> (820 Samples, 13 Neurons)")
print(f"Training Target Set  (y_train) Shape : {y_train.shape} -> (820 Ground-Truth Labels)")
print(f"Testing Input Array  (X_test)  Shape : {X_test.shape} -> (205 Samples, 13 Neurons)")
print(f"Testing Target Set   (y_test)  Shape : {y_test.shape} -> (205 Ground-Truth Labels)")
print(f"=========================================================\n")
