import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    """
    A simple neural network implementation.

    Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output classes.

    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output classes.
        weights_input_hidden (ndarray): Weights connecting input layer to hidden layer.
        biases_input_hidden (ndarray): Biases for the hidden layer.
        weights_hidden_output (ndarray): Weights connecting hidden layer to output layer.
        biases_hidden_output (ndarray): Biases for the output layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.biases_input_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.biases_hidden_output = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """
        Computes the sigmoid activation function.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Computes the softmax activation function.

        Args:
            x (ndarray): Input data.

        Returns:
            ndarray: Output of the softmax function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Performs the forward pass through the neural network.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Predicted probabilities for each class.
        """
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.biases_hidden_output
        self.output = self.softmax(self.output_input)
        
        return self.output

    def backward(self, X, y, output, learning_rate):
        """
        Performs the backward pass through the neural network to update weights and biases.

        Args:
            X (ndarray): Input data.
            y (ndarray): True labels.
            output (ndarray): Predicted probabilities.
            learning_rate (float): Learning rate for updating weights and biases.
        """
        error_output = output - y
        delta_output = error_output

        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.hidden_output * (1 - self.hidden_output)

        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, delta_output)
        self.biases_hidden_output -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        self.weights_input_hidden -= learning_rate * np.dot(X.T, delta_hidden)
        self.biases_input_hidden -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        """
        Trains the neural network.

        Args:
            X (ndarray): Input training data.
            y (ndarray): True labels for the training data.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for updating weights and biases.
        """
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (ndarray): Input data.

        Returns:
            ndarray: Predicted class labels.
        """
        return np.argmax(self.forward(X), axis=1)

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert target labels to one-hot encoding
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 8
output_size = num_classes

# Create and train the neural network
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Predict the class labels for all samples
predicted_labels = model.predict(X)

# Print the predicted class labels for all samples
print("Predicted class labels for all samples:")
for i in range(len(X)):
    print(f"Sample {i+1}: Features = {X[i]}, Predicted Class = {iris.target_names[predicted_labels[i]]}")

# Predict the class labels for the testing set
predicted_labels_test = model.predict(X_test)

# Calculate accuracy on the testing set
accuracy_test = np.mean(predicted_labels_test == np.argmax(y_test, axis=1))
print(f"Accuracy on the testing set: {accuracy_test}")

# Suppose you have a new sample with features [5.1, 3.5, 1.4, 0.2]
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Preprocess the new sample (normalize the features)
new_sample_normalized = scaler.transform(new_sample)

# Predict the class label for the new sample
predicted_label = model.predict(new_sample_normalized)

# Print the predicted class label
print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")

# Suppose you have another new sample with features [5.5, 2.5, 3.8, 1.1]
new_sample = np.array([[5.5, 2.5, 3.8, 1.1]])

# Preprocess the new sample (normalize the features)
new_sample_normalized = scaler.transform(new_sample)

# Predict the class label for the new sample
predicted_label = model.predict(new_sample_normalized)

# Print the predicted class label
print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")

# Suppose you have yet another new sample with features [6.4, 2.8, 5.3, 1.9]
new_sample = np.array([[6.4, 2.8, 5.3, 1.9]])

# Preprocess the new sample (normalize the features)
new_sample_normalized = scaler.transform(new_sample)

# Predict the class label for the new sample
predicted_label = model.predict(new_sample_normalized)

# Print the predicted class label
print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")
