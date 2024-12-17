import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Permute MNIST helper
def permute_mnist(images, seed):
    """Permute the MNIST images using a fixed seed."""
    np.random.seed(seed)
    perm = np.random.permutation(images.shape[1] * images.shape[2])
    permuted_images = images.reshape(-1, images.shape[1] * images.shape[2])[:, perm]
    return permuted_images.reshape(images.shape)

# Fisher Information Matrix class
class EWC:
    def __init__(self, model, lambda_ewc):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.task_weights = []
        self.fisher_information = []

    def compute_fisher_information(self, dataset):
        fisher = [np.zeros_like(w.numpy()) for w in self.model.trainable_weights]
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                preds = self.model(images, training=False)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
            gradients = tape.gradient(loss, self.model.trainable_weights)
            for i, grad in enumerate(gradients):
                fisher[i] += grad.numpy() ** 2
        dataset_size = len(dataset)
        fisher = [f / dataset_size for f in fisher]
        self.fisher_information.append(fisher)

    def store_weights(self):
        self.task_weights.append([w.numpy() for w in self.model.trainable_weights])

    def ewc_loss(self):
        ewc_loss = 0.0
        for task_idx in range(len(self.task_weights)):
            for i, weights in enumerate(self.model.trainable_weights):
                param_diff = weights - self.task_weights[task_idx][i]
                ewc_loss += tf.reduce_sum(self.fisher_information[task_idx][i] * tf.square(param_diff))
        return 0.5 * self.lambda_ewc * ewc_loss

# Prepare MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Experiment settings
num_tasks = 3
batch_size = 128
epochs_per_task = 10
lambda_ewc = 10.0

# Define model architecture
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(400, activation='relu'),
        layers.Dense(400, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Metrics storage
results = {"SGD": [], "L2": [], "EWC": []}

# SGD Baseline
model_sgd = build_model()
for task in range(num_tasks):
    perm_seed = task + 42
    x_train_perm = permute_mnist(x_train, perm_seed)
    x_test_perm = permute_mnist(x_test, perm_seed)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_perm, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_perm, y_test)).batch(batch_size)
    for epoch in range(epochs_per_task):
        model_sgd.fit(train_dataset, epochs=1, verbose=0)
    loss, acc = model_sgd.evaluate(test_dataset, verbose=0)
    results["SGD"].append(acc)

# L2 Regularization
model_l2 = build_model()
l2_lambda = 0.01

for task in range(num_tasks):
    perm_seed = task + 42
    x_train_perm = permute_mnist(x_train, perm_seed)
    x_test_perm = permute_mnist(x_test, perm_seed)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_perm, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_perm, y_test)).batch(batch_size)

    for epoch in range(epochs_per_task):
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                preds = model_l2(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
                # Add L2 Regularization
                l2_loss = l2_lambda * tf.add_n([tf.nn.l2_loss(w) for w in model_l2.trainable_weights])
                total_loss = tf.reduce_mean(loss) + l2_loss
            gradients = tape.gradient(total_loss, model_l2.trainable_weights)
            model_l2.optimizer.apply_gradients(zip(gradients, model_l2.trainable_weights))

    # Evaluate performance on the current task
    loss, acc = model_l2.evaluate(test_dataset, verbose=0)
    results["L2"].append(acc)


# EWC Implementation
model_ewc = build_model()
ewc = EWC(model_ewc, lambda_ewc)
for task in range(num_tasks):
    perm_seed = task + 42
    x_train_perm = permute_mnist(x_train, perm_seed)
    x_test_perm = permute_mnist(x_test, perm_seed)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_perm, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_perm, y_test)).batch(batch_size)
    for epoch in range(epochs_per_task):
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                preds = model_ewc(images, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, preds)
                loss += ewc.ewc_loss()
            gradients = tape.gradient(loss, model_ewc.trainable_weights)
            model_ewc.optimizer.apply_gradients(zip(gradients, model_ewc.trainable_weights))
    ewc.compute_fisher_information(train_dataset)
    ewc.store_weights()
    loss, acc = model_ewc.evaluate(test_dataset, verbose=0)
    results["EWC"].append(acc)

# Plot Results
plt.figure(figsize=(10, 6))
for method, acc in results.items():
    plt.plot(range(1, num_tasks + 1), acc, marker='o', label=method)
plt.xlabel("Task Number")
plt.ylabel("Accuracy")
plt.title("Performance Across Tasks")
plt.legend()
plt.grid(True)
plt.show()
