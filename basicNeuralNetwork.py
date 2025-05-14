import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.biases_input_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.biases_hidden_output = np.zeros((1, output_size))
        
    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):

        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.biases_hidden_output
        self.output = self.softmax(self.output_input)
        
        return self.output

    def backward(self, X, y, output, learning_rate):

        error_output = output - y
        delta_output = error_output

        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.hidden_output * (1 - self.hidden_output)

        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, delta_output)
        self.biases_hidden_output -= learning_rate * np.sum(delta_output, axis=0, keepdims=True)

        self.weights_input_hidden -= learning_rate * np.dot(X.T, delta_hidden)
        self.biases_input_hidden -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):

        return np.argmax(self.forward(X), axis=1)


iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y]

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
hidden_size = 8
output_size = num_classes

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train, y_train, epochs=1000, learning_rate=0.01)

predicted_labels = model.predict(X)

print("Predicted class labels for all samples:")
for i in range(len(X)):
    print(f"Sample {i+1}: Features = {X[i]}, Predicted Class = {iris.target_names[predicted_labels[i]]}")

predicted_labels_test = model.predict(X_test)

accuracy_test = np.mean(predicted_labels_test == np.argmax(y_test, axis=1))
print(f"Accuracy on the testing set: {accuracy_test}")

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])


new_sample_normalized = scaler.transform(new_sample)


predicted_label = model.predict(new_sample_normalized)


print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")


new_sample = np.array([[5.5, 2.5, 3.8, 1.1]])


new_sample_normalized = scaler.transform(new_sample)


predicted_label = model.predict(new_sample_normalized)


print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")

new_sample = np.array([[6.4, 2.8, 5.3, 1.9]])

new_sample_normalized = scaler.transform(new_sample)


predicted_label = model.predict(new_sample_normalized)

print(f"Predicted class label for the new sample: {iris.target_names[predicted_label[0]]}")
