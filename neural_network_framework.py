import numpy as np

class Linear:
    """
    Implements a fully connected (dense) layer.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the layer with weights and biases.
        """
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        self.input = None

    def forward(self, input_data):
        """
        Performs the forward pass.
        """
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        """
        Performs the backward pass and updates the weights and biases.
        """
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input


class ReLU:
    """
    Implements the ReLU activation function.
    """
    def forward(self, input_data):
        """
        Applies the ReLU activation function.
        """
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, grad_output):
        """
        Computes the gradient of the ReLU function.
        """
        return grad_output * (self.input > 0)


class Sigmoid:
    """
    Implements the Sigmoid activation function.
    """
    def forward(self, input_data):
        """
        Applies the Sigmoid activation function.
        """
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
    def backward(self, grad_output):
        """
        Computes the gradient of the Sigmoid function.
        """
        return grad_output * self.output * (1 - self.output)


class Tanh:
    """
    Implements the Tanh activation function.
    """
    def forward(self, input_data):
        """
        Applies the Tanh activation function.
        """
        self.output = np.tanh(input_data)
        return self.output
    
    def backward(self, grad_output):
        """
        Computes the gradient of the Tanh function.
        """
        return grad_output * (1 - self.output ** 2)


class Softmax:
    """
    Implements the Softmax activation function.
    """
    def forward(self, input_data):
        """
        Applies the Softmax activation function.
        """
        exp_scores = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        """
        Computes the gradient of the Softmax function.
        """
        # The derivative is complex; typically handled in combination with cross-entropy loss
        raise NotImplementedError("Backward pass for Softmax is not implemented, typically handled with combined loss")


class CrossEntropyLoss:
    """
    Implements the Cross-Entropy loss function.
    """
    def forward(self, predictions, targets):
        """
        Computes the forward pass of the Cross-Entropy loss.
        """
        self.predictions = predictions
        self.targets = targets
        m = targets.shape[0]
        log_likelihood = -np.log(predictions[range(m), targets])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self):
        """
        Computes the gradient of the Cross-Entropy loss.
        """
        m = self.targets.shape[0]
        grad = self.predictions
        grad[range(m), self.targets] -= 1
        grad /= m
        return grad


class MSELoss:
    """
    Implements the Mean Squared Error loss function.
    """
    def forward(self, predictions, targets):
        """
        Computes the forward pass of the MSE loss.
        """
        self.predictions = predictions
        self.targets = targets
        loss = np.mean((predictions - targets) ** 2)
        return loss

    def backward(self):
        """
        Computes the gradient of the MSE loss.
        """
        return 2 * (self.predictions - self.targets) / self.targets.shape[0]


class SGD:
    """
    Implements the Stochastic Gradient Descent optimizer.
    """
    def __init__(self, learning_rate):
        """
        Initializes the SGD optimizer with a given learning rate.
        """
        self.learning_rate = learning_rate
    
    def step(self, params):
        """
        Updates the parameters using the computed gradients.
        """
        for param in params:
            params[param] -= self.learning_rate * params[param + '_grad']


class Model:
    """
    Implements a neural network model.
    """
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add_layer(self, layer):
        """
        Adds a layer to the model.
        """
        self.layers.append(layer)
    
    def compile(self, loss_function, optimizer):
        """
        Compiles the model with a loss function and optimizer.
        """
        self.loss_function = loss_function
        self.optimizer = optimizer
    
    def forward(self, input_data):
        """
        Performs the forward pass through all layers.
        """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, grad_output):
        """
        Performs the backward pass through all layers.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.optimizer.learning_rate)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Trains the model on the provided data.
        """
        num_samples = X_train.shape[0]
        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                predictions = self.forward(X_batch)
                loss = self.loss_function.forward(predictions, y_batch)
                
                # Backward pass
                grad = self.loss_function.backward()
                self.backward(grad)

                # Update parameters
                for layer in self.layers:
                    if isinstance(layer, Linear):
                        self.optimizer.step({'weights': layer.weights, 'biases': layer.biases})
                
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')
    
    def predict(self, X):
        """
        Predicts the outputs for given inputs.
        """
        return self.forward(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test data.
        """
        predictions = self.predict(X_test)
        loss = self.loss_function.forward(predictions, y_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
        return loss, accuracy

    def save(self, filename):
        """
        Saves the model's weights to a file.
        """
        np.savez(filename, **{f'{layer.__class__.__name__}_weights': layer.weights for layer in self.layers if isinstance(layer, Linear)},
                           **{f'{layer.__class__.__name__}_biases': layer.biases for layer in self.layers if isinstance(layer, Linear)})
    
    def load(self, filename):
        """
        Loads the model's weights from a file.
        """
        data = np.load(filename)
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weights = data[f'{layer.__class__.__name__}_weights']
                layer.biases = data[f'{layer.__class__.__name__}_biases']
from neural_network_framework import Model, CrossEntropyLoss, SGD, Linear, ReLU, Softmax
import numpy as np

# Example usage

# Define a simple neural network using the framework
model = Model()
model.add_layer(Linear(784, 128))
model.add_layer(ReLU())
model.add_layer(Linear(128, 10))
model.add_layer(Softmax())

# Compile the model with loss and optimizer
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)

# Assume x_train, y_train, x_test, y_test are preprocessed and available
# For example purposes, we use random data here
x_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, size=(1000,))
x_test = np.random.randn(200, 784)
y_test = np.random.randint(0, 10, size=(200,))

# Train the model
model.train(x_train, y_train, epochs=20, batch_size=64)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
