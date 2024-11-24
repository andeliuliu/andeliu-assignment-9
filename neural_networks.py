import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Create a results directory
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X  # Store input for backpropagation
        # Hidden layer linear transformation
        self.Z1 = X @ self.W1 + self.b1

        # Activation function
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
        else:
            raise ValueError('Unsupported activation function')

        # Output layer linear transformation
        self.Z2 = self.A1 @ self.W2 + self.b2
        # Sigmoid activation for binary classification
        self.A2 = 1 / (1 + np.exp(-self.Z2))

        return self.A2

    def backward(self, X, y):
        m = y.shape[0]  # Number of samples

        # Compute gradients for output layer
        dZ2 = self.A2 - y  # Derivative of loss w.r.t Z2
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Compute gradients for hidden layer
        dA1 = dZ2 @ self.W2.T
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(self.Z1) ** 2)
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (self.Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-self.Z1))
            dZ1 = dA1 * s * (1 - s)
        else:
            raise ValueError('Unsupported activation function')

        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Store gradients for visualization
        self.dW1 = dW1
        self.dW2 = dW2

# Generate dataset
def generate_data(n_samples=200):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)
    y = y.reshape(-1, 1)
    return X, y

# Visualization function
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(18, 6))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Compute axis limits for hidden features (fixed throughout the animation)
    mlp.forward(X)
    hidden_features = mlp.A1
    x_min, x_max = hidden_features[:, 0].min() - 0.5, hidden_features[:, 0].max() + 0.5
    y_min, y_max = hidden_features[:, 1].min() - 0.5, hidden_features[:, 1].max() + 0.5
    z_min, z_max = hidden_features[:, 2].min() - 0.5, hidden_features[:, 2].max() + 0.5

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                ax_gradient=ax_gradient, X=X, y=y,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max),
        frames=step_num // 10,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

# Update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, x_min, x_max, y_min, y_max, z_min, z_max):
    # Clear previous plots
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform a number of training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    current_step = frame * 10

    # 1. Hidden Space
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Layer Feature Space (Step {current_step})')
    ax_hidden.set_xlim(x_min, x_max)
    ax_hidden.set_ylim(y_min, y_max)
    ax_hidden.set_zlim(z_min, z_max)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    zz = (-mlp.W2[0, 0]*xx - mlp.W2[1, 0]*yy - mlp.b2[0, 0]) / (mlp.W2[2, 0] + 1e-6)
    ax_hidden.plot_surface(xx, yy, zz, alpha=0.3, color='yellow', edgecolor='none')

    # 2. Input Space Decision Boundary
    x_min_input, x_max_input = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min_input, y_max_input = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_input, yy_input = np.meshgrid(np.linspace(x_min_input, x_max_input, 300),
                                     np.linspace(y_min_input, y_max_input, 300))
    grid = np.c_[xx_input.ravel(), yy_input.ravel()]
    probs = mlp.forward(grid).reshape(xx_input.shape)
    ax_input.contourf(xx_input, yy_input, np.ones_like(probs), levels=1, colors=['red'], alpha=0.7)
    ax_input.contour(xx_input, yy_input, probs, levels=[0.5], colors='blue', linewidths=2)
    ax_input.contourf(xx_input, yy_input, probs, levels=[0, 0.5], colors=['lightblue'], alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f'Input Space Decision Boundary (Step {current_step})')

    # 3. Gradients Visualization with Edges
    input_neurons = [(-1, 1), (-1, -1)]
    hidden_neurons = [(0, 2), (0, 0), (0, -2)]
    output_neuron = (1, 0)

    # Plot neurons
    for neuron in input_neurons:
        circle = Circle(neuron, 0.1, color='skyblue')
        ax_gradient.add_patch(circle)
    for neuron in hidden_neurons:
        circle = Circle(neuron, 0.1, color='lightgreen')
        ax_gradient.add_patch(circle)
    circle = Circle(output_neuron, 0.1, color='salmon')
    ax_gradient.add_patch(circle)

    # Plot edges and scale edge thickness by gradient magnitude
    max_grad = max(np.abs(mlp.dW1).max(), np.abs(mlp.dW2).max(), 1e-6)  # Avoid division by zero
    linewidth_scale = 5  # Adjust this value to prevent edges from becoming too thick

    # From input to hidden layer
    for i, input_neuron in enumerate(input_neurons):
        for j, hidden_neuron in enumerate(hidden_neurons):
            grad = mlp.dW1[i, j]
            linewidth = np.clip(np.abs(grad) / max_grad * linewidth_scale, 0.1, linewidth_scale)
            ax_gradient.plot([input_neuron[0], hidden_neuron[0]],
                             [input_neuron[1], hidden_neuron[1]],
                             color='gray', linewidth=linewidth)
    # From hidden to output layer
    for j, hidden_neuron in enumerate(hidden_neurons):
        grad = mlp.dW2[j, 0]
        linewidth = np.clip(np.abs(grad) / max_grad * linewidth_scale, 0.1, linewidth_scale)
        ax_gradient.plot([hidden_neuron[0], output_neuron[0]],
                         [hidden_neuron[1], output_neuron[1]],
                         color='gray', linewidth=linewidth)

    # Labels and settings
    ax_gradient.set_title(f'Network Gradients Visualization (Step {current_step})')
    ax_gradient.set_xlim(-2, 2)
    ax_gradient.set_ylim(-3, 3)
# Main function
if __name__ == "__main__":
    activation = "tanh"  # Change to 'relu' or 'sigmoid' to test other activation functions
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)