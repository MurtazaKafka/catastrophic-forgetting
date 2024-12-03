import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

plot_dir = "ewc_performance_plots"
os.makedirs(plot_dir, exist_ok=True)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

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

class ImprovedElasticWeightConsolidation:
    def __init__(self, model, fisher_multiplier=10000):
        self.model = model
        self.fisher_multiplier = fisher_multiplier
        self.fisher = None
        self.old_params = None

    @tf.function
    def calculate_fisher(self, x, y, num_samples=1000):
        fisher = [tf.zeros_like(v) for v in self.model.trainable_variables]
        for _ in range(num_samples):
            idx = tf.random.uniform([], 0, tf.shape(x)[0], dtype=tf.int32)
            with tf.GradientTape() as tape:
                logits = self.model(tf.expand_dims(x[idx], 0), training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(tf.expand_dims(y[idx], 0), logits)
            grads = tape.gradient(loss, self.model.trainable_variables)
            for i, grad in enumerate(grads):
                fisher[i] += tf.square(grad) / tf.cast(num_samples, tf.float32)
        return fisher

    def store_model_parameters(self):
        self.old_params = [tf.identity(v) for v in self.model.trainable_variables]

    @tf.function
    def ewc_loss(self, y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        if self.fisher is not None and self.old_params is not None:
            for f, p, p_old in zip(self.fisher, self.model.trainable_variables, self.old_params):
                loss += (self.fisher_multiplier / 2) * tf.reduce_sum(f * tf.square(p - p_old))
        return loss

    def train(self, x, y, epochs=5, batch_size=32):
        if self.fisher is None:
            self.fisher = self.calculate_fisher(x, y)
        else:
            new_fisher = self.calculate_fisher(x, y)
            self.fisher = [f1 + f2 for f1, f2 in zip(self.fisher, new_fisher)]

        self.store_model_parameters()

        self.model.compile(optimizer='adam', loss=self.ewc_loss, metrics=['accuracy'])
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

def create_improved_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
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
    plt.savefig(os.path.join(plot_dir, f'{title.replace(" ", "_").lower()}.png'))
    plt.close()

model = create_improved_model()
ewc = ImprovedElasticWeightConsolidation(model)

print("Training on Task 1 (Digits 0-4)")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images_task1, train_labels_task1, epochs=5, validation_split=0.2)

pred_labels_task1, acc_task1 = evaluate_model(model, test_images_task1, test_labels_task1, "Task 1 (Digits 0-4)")
plot_confusion_matrix(test_labels_task1, pred_labels_task1, "Confusion Matrix - Task 1")

print("\nTraining on Task 2 (Digits 5-9) with EWC")
ewc.train(train_images_task2, train_labels_task2, epochs=5)

print("\nEvaluating on both tasks after EWC training:")
pred_labels_all_task1, acc_all_task1 = evaluate_model(model, test_images_task1, test_labels_task1, "Task 1 (Digits 0-4)")
pred_labels_all_task2, acc_all_task2 = evaluate_model(model, test_images_task2, test_labels_task2, "Task 2 (Digits 5-9)")

plot_confusion_matrix(test_labels_task1, pred_labels_all_task1, "Confusion Matrix - Task 1 (After EWC)")
plot_confusion_matrix(test_labels_task2, pred_labels_all_task2, "Confusion Matrix - Task 2 (After EWC)")

test_images_combined = np.concatenate([test_images_task1, test_images_task2])
test_labels_combined = np.concatenate([test_labels_task1, test_labels_task2])

pred_labels_combined, acc_combined = evaluate_model(model, test_images_combined, test_labels_combined, "Combined Tasks")
plot_confusion_matrix(test_labels_combined, pred_labels_combined, "Confusion Matrix - Combined Tasks")

plt.figure(figsize=(12, 6))
accuracies = [acc_task1, acc_all_task1, acc_all_task2, acc_combined]
labels = ['Task 1\nAfter Task 1', 'Task 1\nAfter EWC', 'Task 2\nAfter EWC', 'Combined\nTasks']
colors = ['blue', 'green', 'red', 'purple']

plt.bar(range(len(accuracies)), accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Across Tasks with EWC')
plt.xticks(range(len(accuracies)), labels, rotation=45, ha='right')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'accuracy_comparison_ewc.png'))
plt.close()

print(f"\nAll plots have been saved in the '{plot_dir}' directory.")