import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# CHAPTER 2.4: TRAINING & TESTING SUB-SYSTEM (MEMBER 3 COMPONENT)
# =====================================================================
print("=========================================================")
print("             NEURAL NETWORK TRAINING MATRIX              ")
print("=========================================================")
print(f"Network Topology Layer Config: [{X_train.shape[1]} -> 8 -> 1]")
print("Executing 2,500 Full Gradient Descent Training Epochs...\n")

# Run full batch gradient descent optimization over 2,500 epochs
nn.fit(X_train, y_train, epochs=2500, print_every=500)


print("\n=========================================================")
print("               MODEL TESTING & EVALUATION                ")
print("=========================================================")
print("Evaluating performance across the completely unseen 20% test subset...")

# Pass the held-out test matrix through the optimized network configurations
test_probabilities = nn.forward(X_test)
test_predictions = nn.predict(X_test)

# Flatten matrices to 1D structural vectors for logical comparisons
y_true = y_test.flatten()
y_pred = test_predictions.flatten()

# Calculate Confusion Matrix Quadrants from scratch using logical vectors
TP = np.sum((y_true == 1) & (y_pred == 1)) 
TN = np.sum((y_true == 0) & (y_pred == 0)) 
FP = np.sum((y_true == 0) & (y_pred == 1)) 
FN = np.sum((y_true == 1) & (y_pred == 0)) 

# Calculate statistical evaluations from scratch to maintain framework-free rules
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Track absolute mathematical error margins
mse_loss = np.mean((test_probabilities - y_test) ** 2)
mae_loss = np.mean(np.abs(test_probabilities - y_test))

# Format evaluation metrics table inside terminal console
print("\n+-----------------------------------+--------------------+")
print("| Evaluation Metric                 | Computed Value     |")
print("+-----------------------------------+--------------------+")
print(f"| Test Classification Accuracy      | {accuracy * 100:16.2f}% |")
print(f"| Diagnostic Precision Score        | {precision * 100:16.2f}% |")
print(f"| Diagnostic Recall (Sensitivity)   | {recall * 100:16.2f}% |")
print(f"| Mean Squared Error (MSE Loss)     | {mse_loss:18.5f} |")
print(f"| Mean Absolute Error (MAE Loss)    | {mae_loss:18.5f} |")
print("+-----------------------------------+--------------------+")

print(f"\nCaptured Confusion Matrix Quadrants:")
print(f"-> True Negatives  (TN): {TN:3d}  |  False Positives (FP): {FP:3d}")
print(f"-> False Negatives (FN): {FN:3d}  |  True Positives  (TP): {TP:3d}")
print("=========================================================\n")

# Generate Training History Curves for Report Analysis
plt.figure(figsize=(12, 5))

# Plot 1: Mean Squared Error Loss Drop Progression Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, len(nn.loss_history) + 1), nn.loss_history, color='crimson', lw=2)
plt.title('Training Loss Progression', fontsize=12, fontweight='bold')
plt.xlabel('Epochs Count', fontsize=10)
plt.ylabel('Mean Squared Error (MSE)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Accuracy Improvements Progression Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, len(nn.acc_history) + 1), np.array(nn.acc_history) * 100, color='teal', lw=2)
plt.title('Training Accuracy Progression', fontsize=12, fontweight='bold')
plt.xlabel('Epochs Count', fontsize=10)
plt.ylabel('Model Accuracy Percentage (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
