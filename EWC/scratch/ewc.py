import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

def split_data(images, labels, task):
    if task == 1:
        mask = labels < 5
    else:
        mask = labels >= 5
    return images[mask], labels[mask]

train_images_task1, train_labels_task1 = split_data(train_images, train_labels, 1)
train_images_task2, train_labels_task2 = split_data(train_images, train_labels, 2)
test_images_task1, test_labels_task1 = split_data(test_images, test_labels, 1)
test_images_task2, test_labels_task2 = split_data(test_images, test_labels, 2)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
    return model

def evaluate_model(model, images, labels, task_name):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_labels)
    print(f"\nPerformance on {task_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, predicted_labels))
    return predicted_labels, accuracy

def plot_confusion_matrix(true_labels, predicted_labels, title):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.close()

class ElasticWeightConsolidation:
    def __init__(self, model, fisher_multiplier=1000):
        self.model = model
        self.fisher_multiplier = fisher_multiplier
        self.fisher = None
        self.old_params = None

    def calculate_fisher(self, x, y, num_samples=1000):
        fisher = [tf.zeros_like(v) for v in self.model.trainable_variables]
        for _ in range(num_samples):
            idx = np.random.randint(0, len(x))
            with tf.GradientTape() as tape:
                logits = self.model(tf.expand_dims(x[idx], 0), training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(y[idx], 0), logits)
            grads = tape.gradient(loss, self.model.trainable_variables)
            for i, grad in enumerate(grads):
                fisher[i] += grad ** 2 / num_samples
        return fisher

    def store_model_parameters(self):
        self.old_params = [tf.identity(v) for v in self.model.trainable_variables]

    def ewc_loss(self, x, y):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, self.model(x, training=True))
        if self.fisher is not None and self.old_params is not None:
            for f, p, p_old in zip(self.fisher, self.model.trainable_variables, self.old_params):
                loss += (self.fisher_multiplier / 2) * tf.reduce_sum(f * (p - p_old) ** 2)
        return loss

    def train(self, x, y, epochs=5, batch_size=32):
        if self.fisher is None:
            self.fisher = self.calculate_fisher(x, y)
        else:
            new_fisher = self.calculate_fisher(x, y)
            self.fisher = [f1 + f2 for f1, f2 in zip(self.fisher, new_fisher)]

        self.store_model_parameters()

        optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                loss = self.ewc_loss(x_batch, y_batch)
            grads = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            num_batches = 0
            for x_batch, y_batch in dataset:
                loss = train_step(x_batch, y_batch)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Average loss: {avg_loss:.4f}")

def adjust_labels(labels, task):
    if task == 1:
        return labels
    else:
        return labels - 5

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

ewc = ElasticWeightConsolidation(model)

train_labels_task1_adjusted = adjust_labels(train_labels_task1, 1)
test_labels_task1_adjusted = adjust_labels(test_labels_task1, 1)
ewc.train(train_images_task1, train_labels_task1_adjusted, epochs=5)
pred_labels_task1, acc_task1 = evaluate_model(model, test_images_task1, test_labels_task1_adjusted, "Task 1 (Digits 0-4)")
plot_confusion_matrix(test_labels_task1_adjusted, pred_labels_task1, "Confusion Matrix - Task 1")

train_labels_task2_adjusted = adjust_labels(train_labels_task2, 2)
test_labels_task2_adjusted = adjust_labels(test_labels_task2, 2)
ewc.train(train_images_task2, train_labels_task2_adjusted, epochs=5)

print("\nEvaluating on both tasks after EWC training:")
pred_labels_all_task1, acc_all_task1 = evaluate_model(model, test_images_task1, test_labels_task1_adjusted, "Task 1 (Digits 0-4)")
pred_labels_all_task2, acc_all_task2 = evaluate_model(model, test_images_task2, test_labels_task2_adjusted, "Task 2 (Digits 5-9)")

plot_confusion_matrix(test_labels_task1_adjusted, pred_labels_all_task1, "Confusion Matrix - Task 1 (After EWC)")
plot_confusion_matrix(test_labels_task2_adjusted, pred_labels_all_task2, "Confusion Matrix - Task 2 (After EWC)")

pred_labels_all = np.concatenate([pred_labels_all_task1, pred_labels_all_task2 + 5])
test_labels_all = np.concatenate([test_labels_task1, test_labels_task2])

plot_confusion_matrix(test_labels_all, pred_labels_all, "Confusion Matrix - All Tasks (After EWC)")

plt.figure(figsize=(12, 6))
accuracies = [acc_task1, acc_all_task1, acc_all_task2]
labels = ['Task 1\nAfter Task 1', 'Task 1\nAfter EWC', 'Task 2\nAfter EWC']
colors = ['blue', 'green', 'red']

plt.bar(range(len(accuracies)), accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Across Tasks with EWC')
plt.xticks(range(len(accuracies)), labels, rotation=45, ha='right')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('accuracy_comparison_ewc.png')
plt.close()

print("\nAccuracy comparison plot saved as 'accuracy_comparison_ewc.png'")