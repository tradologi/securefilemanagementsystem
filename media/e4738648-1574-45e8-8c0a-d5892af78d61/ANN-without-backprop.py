import numpy as np


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        self.hidden = self.sigmoid(
            np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(
            np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return self.output

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            output = self.forward(X)

            error = y - output

            d_output = error * output * (1 - output)
            d_hidden = np.dot(
                d_output, self.weights_hidden_output.T) * self.hidden * (1 - self.hidden)

            self.weights_hidden_output += learning_rate * \
                np.dot(self.hidden.T, d_output)
            self.bias_output += learning_rate * \
                np.sum(d_output, axis=0, keepdims=True)
            self.weights_input_hidden += learning_rate * np.dot(X.T, d_hidden)
            self.bias_hidden += learning_rate * \
                np.sum(d_hidden, axis=0, keepdims=True)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    nn.train(X, y, learning_rate=0.1, epochs=10000)

    for i in range(len(X)):
        prediction = nn.forward(X[i:i+1])
        print(
            f"Input: {X[i]}, Target: {y[i]}, Prediction: {prediction[0][0]:.4f}")
