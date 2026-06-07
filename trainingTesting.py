import numpy as np


input_neurons = X_train.shape[1]   # Evaluates to 13
hidden_neurons = 8                 # Configured exactly to project layout
output_neurons = 1                 # Single classification output channel

nn = SimpleNeuralNetwork(input_neurons, hidden_neurons, output_neurons, lr=0.6)

print("=========================================================")
print("             NEURAL NETWORK TRAINING MATRIX              ")
print("=========================================================")
print(f"Network Topology Layer Config: [{input_neurons} -> {hidden_neurons} -> {output_neurons}]")
print("Executing 2,500 Full Gradient Descent Training Epochs...\n")

nn.fit(X_train, y_train, epochs=2500, print_every=500)


print("\n=========================================================")
print("               MODEL TESTING & EVALUATION                ")
print("=========================================================")
print("Evaluating performance across the completely unseen 20% test subset...")


test_probabilities = nn.forward(X_test)
# Convert probabilities to definitive binary classifications (0 or 1) using 0.5 threshold
test_predictions = nn.predict(X_test)

y_true = y_test.flatten()
y_pred = test_predictions.flatten()


TP = np.sum((y_true == 1) & (y_pred == 1)) 
TN = np.sum((y_true == 0) & (y_pred == 0)) # True Temporaries: Correctly identified low risk
FP = np.sum((y_true == 0) & (y_pred == 1)) # False Positives: Healthy classified as high risk
FN = np.sum((y_true == 1) & (y_pred == 0)) # False Negatives: Sick classified as low risk


accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0


mse_loss = np.mean((test_probabilities - y_test) ** 2)
mae_loss = np.mean(np.abs(test_probabilities - y_test))


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
print(f"-> True Positives  (TP): {TP:3d}  |  False Positives (FP): {FP:3d}")
print(f"-> False Negatives (FN): {FN:3d}  |  True Negatives  (TN): {TN:3d}")
print("=========================================================\n")

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(range(1, len(nn.loss_history) + 1), nn.loss_history, color='crimson', lw=2)
plt.title('Training Loss Progression', fontsize=12, fontweight='bold')
plt.xlabel('Epochs Count', fontsize=10)
plt.ylabel('Mean Squared Error (MSE)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)


plt.subplot(1, 2, 2)
plt.plot(range(1, len(nn.acc_history) + 1), np.array(nn.acc_history) * 100, color='teal', lw=2)
plt.title('Training Accuracy Progression', fontsize=12, fontweight='bold')
plt.xlabel('Epochs Count', fontsize=10)
plt.ylabel('Model Accuracy Percentage (%)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
