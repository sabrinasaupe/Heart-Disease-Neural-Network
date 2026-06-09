import numpy as np


def sigmoid(z):
    # keep outputs between 0 and 1
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# network setup
input_neurons = 13
hidden_neurons = 8
output_neurons = 1

# start with simple values for testing
W1 = np.full((input_neurons, hidden_neurons), 0.1)
W2 = np.full((hidden_neurons, output_neurons), 0.1)

b1 = np.zeros((1, hidden_neurons))
b2 = np.zeros((1, output_neurons))

learning_rate = 0.01


def forward_propagation(X):

    # hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # final prediction
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2


def compute_loss(y_true, y_pred):
    # Mean Squared Error formulation
    return np.mean((y_true - y_pred) ** 2)


def backward_propagation(X, y, Z1, A1, Z2, A2):
    m = X.shape[0]

    # CORRECTED OUTPUT LAYER GRADIENT FOR MSE + SIGMOID
    # dZ2 = 2/m * (A2 - y) * sigmoid_derivative(A2)
    dZ2 = (2 / m) * (A2 - y) * (A2 * (1 - A2))
    
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Propagate error back to hidden layer (remains structurally consistent)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (A1 * (1 - A1))

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def update_parameters(dW1, db1, dW2, db2):

    global W1, b1, W2, b2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


def train(X, y, epochs):

    for epoch in range(epochs):

        Z1, A1, Z2, A2 = forward_propagation(X)

        loss = compute_loss(y, A2)

        dW1, db1, dW2, db2 = backward_propagation(
            X, y, Z1, A1, Z2, A2
        )

        update_parameters(dW1, db1, dW2, db2)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")


def predict(X):
    _, _, _, A2 = forward_propagation(X)
    binary = (A2 >= 0.5).astype(int)
    return A2, binary  # returns (confidence score, 0 or 1)
