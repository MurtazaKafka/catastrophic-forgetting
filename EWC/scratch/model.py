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
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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

model = create_model()
history_task1 = model.fit(train_images_task1, train_labels_task1, epochs=5, validation_split=0.2, verbose=1)
pred_labels_task1, acc_task1 = evaluate_model(model, test_images_task1, test_labels_task1, "Task 1 (Digits 0-4)")

plot_confusion_matrix(test_labels_task1, pred_labels_task1, "Confusion Matrix - Task 1")

model.save('task1_model.keras')

history_task2 = model.fit(train_images_task2, train_labels_task2, epochs=5, validation_split=0.2, verbose=1)
pred_labels_task2, acc_task2 = evaluate_model(model, test_images_task2, test_labels_task2, "Task 2 (Digits 5-9)")

plot_confusion_matrix(test_labels_task2, pred_labels_task2, "Confusion Matrix - Task 2")

# Evaluate on both tasks after training on Task 2
print("\nEvaluating on both tasks after training on Task 2:")
pred_labels_all_task1, acc_all_task1 = evaluate_model(model, test_images_task1, test_labels_task1, "Task 1 (Digits 0-4)")
pred_labels_all_task2, acc_all_task2 = evaluate_model(model, test_images_task2, test_labels_task2, "Task 2 (Digits 5-9)")

pred_labels_all = np.concatenate([pred_labels_all_task1, pred_labels_all_task2])
test_labels_all = np.concatenate([test_labels_task1, test_labels_task2])

plot_confusion_matrix(test_labels_all, pred_labels_all, "Confusion Matrix - All Tasks")

plt.figure(figsize=(12, 6))
accuracies = [acc_task1, acc_task2, acc_all_task1, acc_all_task2]
labels = ['Task 1\nAfter Task 1', 'Task 2\nAfter Task 2', 'Task 1\nAfter Task 2', 'Task 2\nAfter Task 2']
colors = ['blue', 'green', 'red', 'orange']

plt.bar(range(len(accuracies)), accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Across Tasks')
plt.xticks(range(len(accuracies)), labels, rotation=45, ha='right')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
plt.close()

print("\nAccuracy comparison plot saved as 'accuracy_comparison.png'")

task1_model = tf.keras.models.load_model('task1_model.keras')
print("\nEvaluating with Task 1 model:")
evaluate_model(task1_model, test_images_task1, test_labels_task1, "Task 1 (Digits 0-4)")
evaluate_model(task1_model, test_images_task2, test_labels_task2, "Task 2 (Digits 5-9)")