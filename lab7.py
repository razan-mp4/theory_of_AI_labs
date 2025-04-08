import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix, classification_report

# 1. Завантаження та підготовка даних
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 2. Побудова архітектури (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Компіляція
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Навчання (збережемо історію для графіків)
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_split=0.1,
    batch_size=64
)

# 5. Графіки Loss та Accuracy
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss by Epoch')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy by Epoch')
plt.legend()
plt.show()

# 6. Оцінка на тестових даних
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Точність на тестових даних: {test_acc * 100:.2f}%')

# 7. Передбачення та візуалізація на окремому зображенні
image_index = 0
prediction = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
predicted_class = np.argmax(prediction)

print(f'Модель вважає, що це цифра: {predicted_class}')
print(f'Справжня цифра: {y_test[image_index]}')

plt.imshow(x_test[image_index].reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {predicted_class}, True: {y_test[image_index]}')
plt.axis('off')
plt.show()

# 8. Побудова матриці похибок (Confusion Matrix)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

# Відобразимо у вигляді теплої карти (matshow)
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Додатково можна подивитись деталізований звіт за кожним класом
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# 9. Гістограма передбачених класів
plt.hist(y_pred_classes, bins=10, rwidth=0.8)
plt.title('Histogram of Predicted Classes (0-9)')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.show()
