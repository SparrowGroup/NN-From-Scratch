# **Evaluating on the Test Set**
evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.0001, dropout_rate=0.2)
default:~/NN-From-Scratch$ cat main.py
import numpy as np
import cupy as cp  # GPU-accelerated computing with CuPy
import pandas as pd
from matplotlib import pyplot as plt
import os
import urllib.request
from tqdm import tqdm
import tarfile
import pickle
import time

# CIFAR-100 dataset URL and filename
cifar100_url = "https:gwww.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
cifar100_filename = "cifar-100-python.tar.gz"

# Directory to save downloaded and extracted data
data_dir = "./data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def download_with_progress(url, filename):
    """Download a file from a URL with a progress bar."""
    response = urllib.request.urlopen(url)
    file_size = int(response.info().get('Content-Length', -1))

    with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as t:
        def reporthook(block_num, block_size, _):
            t.update(block_size)

        urllib.request.urlretrieve(url, filename, reporthook)


def download_cifar100():
    """Download and extract the CIFAR-100 dataset."""
    file_path = os.path.join(data_dir, cifar100_filename)
    extracted_dir = os.path.join(data_dir, 'cifar-100-python')

    if not os.path.exists(file_path):
        print(f"Downloading {cifar100_filename}...")
        download_with_progress(cifar100_url, file_path)
    else:
        print(f"{cifar100_filename} is already downloaded! Woohoo!")

    # Extract the dataset if not already extracted
    if not os.path.exists(extracted_dir):
        print('Extracting data...')
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print('Data extracted successfully!')
    else:
        print('Data is already extracted!')


def unpickle(file):
    """Load a pickled CIFAR-100 batch."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch(file):
    """Load data and labels from a CIFAR-100 batch file."""
    batch = unpickle(file)
    X = batch[b'data']
    y = np.array(batch[b'fine_labels'])
    return X, y


def load_data(dir):
    """Load training and test data from CIFAR-100."""
    train_file = os.path.join(data_dir, 'cifar-100-python', 'train')
    test_file = os.path.join(data_dir, 'cifar-100-python', 'test')

    X_train, y_train = load_batch(train_file)
    X_test, y_test = load_batch(test_file)

    return X_train, y_train, X_test, y_test


def show_random_samples(X_train, y_train, num_samples=3):
    """Display random samples from the training set."""
    X_train_cpu = cp.asnumpy(X_train)
    y_train_cpu = cp.asnumpy(y_train)

    random_indices = np.random.choice(X_train_cpu.shape[1], num_samples, replace=False)

    for idx in random_indices:
        image = X_train_cpu[:, idx].reshape(3, 32, 32).transpose(1, 2, 0)
        label = y_train_cpu[idx]

        print(f'Label: {label}')
        plt.imshow(image)
        plt.show()


def standardize_data(X_train, X_test):
    """Standardize the dataset to have zero mean and unit variance per feature."""
    mean = cp.mean(X_train, axis=1, keepdims=True)  # Shape: (3072, 1)
    std = cp.std(X_train, axis=1, keepdims=True) + 1e-8  # Shape: (3072, 1)
    X_train_standardized = (X_train - mean) / std
    X_test_standardized = (X_test - mean) / std
    return X_train_standardized, X_test_standardized


# Download and load the data
download_cifar100()
X_train, y_train, X_test, y_test = load_data(data_dir)

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Training data shape:", X_train.shape)  # (50000, 3072)
print("Training labels shape:", y_train.shape)  # (50000,)
print("Test data shape:", X_test.shape)  # (10000, 3072)
print("Test labels shape:", y_test.shape)  # (10000,)

# Transfer data to GPU and transpose to (features, samples)
X_train_gpu = cp.asarray(X_train.T)  # Shape: (3072, 50000)
y_train_gpu = cp.asarray(y_train)
X_test_gpu = cp.asarray(X_test.T)  # Shape: (3072, 10000)
y_test_gpu = cp.asarray(y_test)

# Display random samples to verify data
show_random_samples(X_train_gpu, y_train_gpu)

# Standardize the data
X_train_gpu, X_test_gpu = standardize_data(X_train_gpu, X_test_gpu)

print("Data standardized.")


def init_params(layers):
    """Initialize network parameters with He initialization."""
    params = {}
    for l in range(1, len(layers)):
        params[f'W{l}'] = cp.random.randn(layers[l], layers[l - 1]) * cp.sqrt(2. / layers[l - 1])
        params[f'b{l}'] = cp.zeros((layers[l], 1))
    return params


def ReLU(Z):
    """ReLU activation function."""
    return cp.maximum(0, Z)


def Leaky_ReLU(Z, alpha=0.01):
    """Leaky ReLU activation function."""
    return cp.where(Z > 0, Z, alpha * Z)


def softmax(Z):
    """Softmax activation function with numerical stability."""
    shift_Z = Z - cp.max(Z, axis=0, keepdims=True)
    exp_Z = cp.exp(shift_Z)
    return exp_Z / cp.sum(exp_Z, axis=0, keepdims=True)


def batch_norm_forward(Z, gamma, beta, eps=1e-5):
    """Forward pass for batch normalization."""
    mu = cp.mean(Z, axis=1, keepdims=True)
    var = cp.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mu) / cp.sqrt(var + eps)
    out = gamma * Z_norm + beta
    cache = (Z, Z_norm, mu, var, gamma, beta, eps)
    return out, cache


def batch_norm_backward(dout, cache):
    """Backward pass for batch normalization."""
    Z, Z_norm, mu, var, gamma, beta, eps = cache
    m = Z.shape[1]

    dgamma = cp.sum(dout * Z_norm, axis=1, keepdims=True)
    dbeta = cp.sum(dout, axis=1, keepdims=True)

    dZ_norm = dout * gamma
    dvar = cp.sum(dZ_norm * (Z - mu) * -0.5 * (var + eps) ** (-1.5), axis=1, keepdims=True)
    dmu = cp.sum(dZ_norm * -1 / cp.sqrt(var + eps), axis=1, keepdims=True) + dvar * cp.mean(-2 * (Z - mu), axis=1,
                                                                                            keepdims=True)
    dZ = dZ_norm / cp.sqrt(var + eps) + dvar * 2 * (Z - mu) / m + dmu / m

    return dZ, dgamma, dbeta


def forward_prop(params, X, training=True, dropout_rate=0.5):
    """Perform forward propagation with batch normalization and dropout."""
    caches = {}
    A = X
    # Correctly determine the number of layers based on W parameters
    L = len([key for key in params.keys() if key.startswith('W')])

    for l in range(1, L):
        Z = params[f'W{l}'] @ A + params[f'b{l}']

        # Batch Normalization
        if f'gamma{l}' in params and f'beta{l}' in params:
            Z, cache_bn = batch_norm_forward(Z, params[f'gamma{l}'], params[f'beta{l}'])
            caches[f'bn{l}'] = cache_bn

        A = Leaky_ReLU(Z)
        caches[f'Z{l}'] = Z
        caches[f'A{l}'] = A

        if training:
            D = (cp.random.rand(*A.shape) > dropout_rate).astype(cp.float32)
            A *= D
            A /= (1.0 - dropout_rate)
            caches[f'D{l}'] = D

    # Output layer
    Z = params[f'W{L}'] @ A + params[f'b{L}']

    caches[f'Z{L}'] = Z
    A_final = softmax(Z)
    caches[f'A{L}'] = A_final

    return A_final, caches


def one_hot(Y, num_classes=100):
    """Convert labels to one-hot encoding."""
    one_hot_Y = cp.zeros((num_classes, Y.size), dtype=cp.float32)
    one_hot_Y[Y, cp.arange(Y.size)] = 1
    return one_hot_Y


def cross_entropy_loss(A3, Y, params, lambda_reg=0.001):
    """Compute cross-entropy loss with L2 regularization."""
    m = Y.size

    # Clip A3 to prevent log(0)
    A3_clipped = cp.clip(A3, 1e-10, 1.0)

    # Compute cross-entropy loss
    log_probs = -cp.log(A3_clipped[Y, cp.arange(m)])
    loss = cp.sum(log_probs) / m

    # L2 regularization
    l2_loss = 0

    # Identify all weight matrices (keys starting with 'W')
    weight_keys = [key for key in params if key.startswith('W')]
    num_weights = len(weight_keys)
    # print(f"Number of weight matrices detected: {num_weights}")

    for key in weight_keys:
        l2 = cp.sum(params[key] ** 2)
        l2_loss += l2
        # print(f"L2 loss for {key}: {l2.get():.4f}")

    l2_loss = (lambda_reg / (2 * m)) * l2_loss
    total_loss = loss + l2_loss

    # print(f"Total cross-entropy loss: {loss.get():.4f}")
    # print(f"Total L2 regularization loss: {l2_loss.get():.4f}")
    # print(f"Combined total loss: {total_loss.get():.4f}")

    return total_loss


def deriv_Leaky_ReLU(Z, alpha=0.01):
    """Derivative of Leaky ReLU activation."""
    return cp.where(Z > 0, 1, alpha)


def backward_prop(params, caches, X, Y, lambda_reg=0.001, dropout_rate=0.5):
    """Perform backpropagation to compute gradients with batch normalization and dropout."""
    grads = {}
    # Calculate L based on the number of W parameters
    L = len([key for key in params.keys() if key.startswith('W')])
    m = Y.size
    one_hot_Y = one_hot(Y)

    # Output layer gradients
    A_final = caches[f'A{L}']
    dZ = A_final - one_hot_Y  # Shape: (100, m)
    grads[f'dW{L}'] = (dZ @ caches[f'A{L - 1}'].T) / m + (lambda_reg / m) * params[f'W{L}']
    grads[f'db{L}'] = cp.sum(dZ, axis=1, keepdims=True) / m

    # Backpropagate through layers L-1 to 1
    dA_prev = params[f'W{L}'].T @ dZ

    for l in range(L - 1, 0, -1):
        # Dropout
        if f'D{l}' in caches:
            dA_prev *= caches[f'D{l}']
            dA_prev /= (1.0 - dropout_rate)

        # Activation derivative
        dZ = dA_prev * deriv_Leaky_ReLU(caches[f'Z{l}'])

        # Batch Normalization backward
        if f'bn{l}' in caches:
            dZ, dgamma, dbeta = batch_norm_backward(dZ, caches[f'bn{l}'])
            grads[f'dgamma{l}'] = cp.sum(dgamma, axis=1, keepdims=True) / m
            grads[f'dbeta{l}'] = cp.sum(dbeta, axis=1, keepdims=True) / m

        # Gradients
        A_prev = X if l == 1 else caches[f'A{l - 1}']
        grads[f'dW{l}'] = (dZ @ A_prev.T) / m + (lambda_reg / m) * params[f'W{l}']
        grads[f'db{l}'] = cp.sum(dZ, axis=1, keepdims=True) / m

        if l > 1:
            dA_prev = params[f'W{l}'].T @ dZ

    return grads


def update_params(params, grads, optimizer_state, alpha, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
    """Update network parameters using Adam optimizer."""
    # Correctly determine the number of layers based on W parameters
    L = len([key for key in params.keys() if key.startswith('W')])

    for l in range(1, L + 1):
        # Initialize state if not present
        if f'mW{l}' not in optimizer_state:
            optimizer_state[f'mW{l}'] = cp.zeros_like(params[f'W{l}'])
            optimizer_state[f'vW{l}'] = cp.zeros_like(params[f'W{l}'])
            optimizer_state[f'mb{l}'] = cp.zeros_like(params[f'b{l}'])
            optimizer_state[f'vb{l}'] = cp.zeros_like(params[f'b{l}'])

        # Update first moment estimate
        optimizer_state[f'mW{l}'] = beta1 * optimizer_state[f'mW{l}'] + (1 - beta1) * grads[f'dW{l}']
        optimizer_state[f'mb{l}'] = beta1 * optimizer_state[f'mb{l}'] + (1 - beta1) * grads[f'db{l}']
        # Update second moment estimate
        optimizer_state[f'vW{l}'] = beta2 * optimizer_state[f'vW{l}'] + (1 - beta2) * (grads[f'dW{l}'] ** 2)
        optimizer_state[f'vb{l}'] = beta2 * optimizer_state[f'vb{l}'] + (1 - beta2) * (grads[f'db{l}'] ** 2)
        # Compute bias-corrected moment estimates
        mW_corrected = optimizer_state[f'mW{l}'] / (1 - beta1 ** t)
        mb_corrected = optimizer_state[f'mb{l}'] / (1 - beta1 ** t)
        vW_corrected = optimizer_state[f'vW{l}'] / (1 - beta2 ** t)
        vb_corrected = optimizer_state[f'vb{l}'] / (1 - beta2 ** t)
        # Update parameters
        params[f'W{l}'] -= alpha * mW_corrected / (cp.sqrt(vW_corrected) + epsilon)
        params[f'b{l}'] -= alpha * mb_corrected / (cp.sqrt(vb_corrected) + epsilon)
    return params, optimizer_state

def get_predictions(A3):
    """Get class predictions from output probabilities."""
    return cp.argmax(A3, axis=0)


def get_accuracy(predictions, Y):
    """Compute accuracy of predictions."""
    return cp.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha, batch_size=256, lambda_reg=0.001, dropout_rate=0.5):
    """Train the network using Adam optimizer with mini-batches, batch normalization, and dropout."""

    layers = [3072, 2048, 1024, 512, 256, 100]  # Define network architecture
    params = init_params(layers)

    # Initialize batch normalization parameters
    for l in range(1, len(layers)):
        params[f'gamma{l}'] = cp.ones((layers[l], 1), dtype=cp.float32)
        params[f'beta{l}'] = cp.zeros((layers[l], 1), dtype=cp.float32)

    optimizer_state = {}
    m = X.shape[1]
    t = 1  # Time step for Adam

    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for i in range(1, iterations + 1):
        # Shuffle data at the start of each epoch
        permutation = cp.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        # Process mini-batches
        for j in range(0, m, batch_size):
            end_j = min(j + batch_size, m)
            X_batch = X_shuffled[:, j:end_j]
            Y_batch = Y_shuffled[j:end_j]

            # Forward propagation with dropout and batch normalization
            A_final, caches = forward_prop(params, X_batch, training=True, dropout_rate=dropout_rate)

            # Backward propagation
            grads = backward_prop(params, caches, X_batch, Y_batch, lambda_reg=lambda_reg, dropout_rate=dropout_rate)

            # Update parameters using Adam optimizer
            params, optimizer_state = update_params(params, grads, optimizer_state, alpha, t=t)
            t += 1

        # Compute training loss and accuracy at the end of the epoch
        A_train, _ = forward_prop(params, X, training=False, dropout_rate=dropout_rate)
        loss = cross_entropy_loss(A_train, Y, params, lambda_reg)
        predictions = get_predictions(A_train)
        accuracy = get_accuracy(predictions, Y)

        # Early stopping check
        if loss.get() < best_loss:
            best_loss = loss.get()
            patience_counter = 0
            best_params = {k: v.copy() for k, v in params.items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_params


def evaluate(params, X, Y, lambda_reg=0.001, dropout_rate=0.5):
    """Evaluate the trained model on a given dataset without dropout."""
    A_final, _ = forward_prop(
        params, X,
        dropout_rate=dropout_rate,
        training=False
    )
    predictions = get_predictions(A_final)
    accuracy = get_accuracy(predictions, Y)
    loss = cross_entropy_loss(A_final, Y, params, lambda_reg)
    print(f"Evaluation:")
    print(f"Accuracy: {accuracy.get() * 100:.2f}%")
    print(f"Loss: {loss.get():.4f}\n")


# **Training the Model on the Full Dataset with Batch Normalization and Dropout**
print("Starting training with enhanced MLP...")
W1, b1, W2, b2, W3, b3 = None, None, None, None, None, None  # Placeholder if needed

best_params = gradient_descent(
    X_train_gpu,
    y_train_gpu,
    iterations=600,
    alpha=0.001,  # Lower learning rate for Adam
    batch_size=256,
    lambda_reg=0.001,  # Reduced regularization strength
    dropout_rate=0.3  # 50% dropout rate
)
print("600 iter mid-low learning rate, low dropout training completed.")

# **Evaluating on the Test Set**
evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.005, dropout_rate=0.3)

best_params = gradient_descent(
    X_train_gpu,
    y_train_gpu,
    iterations=600,
    alpha=0.001,  # Lower learning rate for Adam
    batch_size=256,
    lambda_reg=0.005,  # Reduced regularization strength
    dropout_rate=0.25  # 50% dropout rate
)
print("600 iter model 2 training completed.")

# **Evaluating on the Test Set**
evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.005, dropout_rate=0.25)

best_params = gradient_descent(
    X_train_gpu,
    y_train_gpu,
    iterations=800,
    alpha=0.001,  # Lower learning rate for Adam
    batch_size=128,
    lambda_reg=0.01,  # Reduced regularization strength
    dropout_rate=0.3  # 50% dropout rate
)
print("800 iter model training completed.")

# **Evaluating on the Test Set**
evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.0001, dropout_rate=0.5)

# best_params = gradient_descent(
#    X_train_gpu,
#    y_train_gpu,
#    iterations=600,
#    alpha=0.001,  # Lower learning rate for Adam
#    batch_size=256,
#    lambda_reg=0.01,  # Reduced regularization strength
#    dropout_rate=0.3  # 50% dropout rate
# )
# print("600 iter low dropout rate training completed.")

# **Evaluating on the Test Set**
# evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.01, dropout_rate=0.3)

# best_params = gradient_descent(
#    X_train_gpu,
#    y_train_gpu,
#    iterations=600,
#    alpha=0.001,  # Lower learning rate for Adam
#    batch_size=512,
#    lambda_reg=0.01,  # Reduced regularization strength
#    dropout_rate=0.5  # 50% dropout rate
# )
# print("600 iter large batch training completed.")

# **Evaluating on the Test Set**
# evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.0001, dropout_rate=0.5)

best_params = gradient_descent(
    X_train_gpu,
    y_train_gpu,
    iterations=600,
    alpha=0.001,  # Lower learning rate for Adam
    batch_size=256,
    lambda_reg=0.01,  # Reduced regularization strength
    dropout_rate=0.2  # 50% dropout rate
)
print("600 iter low dropout training completed.")

# **Evaluating on the Test Set**
evaluate(best_params, X_test_gpu, y_test_gpu, lambda_reg=0.0001, dropout_rate=0.2)