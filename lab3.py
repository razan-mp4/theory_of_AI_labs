import numpy as np
import matplotlib.pyplot as plt

# Функція активації (порогова)
def step_function(x):
    return 1 if x >= 0 else 0

# Функція обчислення втрат (Loss) – Бінарна крос-ентропія
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Запобігання log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Визначаємо вагові коефіцієнти та пороги для логічних функцій
# Ваги та поріг для "АБО" (OR)
w_or = np.array([1, 1])
b_or = -0.5  # Поріг

# Ваги та поріг для "І" (AND)
w_and = np.array([1, 1])
b_and = -1.5  # Поріг

# Вхідні комбінації
inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

# Очікувані результати
expected_output = np.array([0, 0, 1, 1])

# Кількість епох для тренування
epochs = 10
loss_values = []
accuracy_values = []

for epoch in range(epochs):
    # Обчислення виходів нейрона "АБО"
    output_or = np.array([step_function(np.dot(x, w_or) + b_or) for x in inputs])

    # Формування нового входу для нейрона "І"
    inputs_and = np.column_stack((inputs[:, 0], output_or))

    # Обчислення виходів нейрона "І"
    output_final = np.array([step_function(np.dot(x, w_and) + b_and) for x in inputs_and])

    # Обчислення втрат та точності
    loss = binary_cross_entropy(expected_output, output_final)
    accuracy = np.mean(output_final == expected_output)
    loss_values.append(loss)
    accuracy_values.append(accuracy)

# Виведення результатів
print("Вхідні дані (x1, x2) -> OR -> AND -> Вихід (очікуваний)")
for i in range(len(inputs)):
    print(f"{inputs[i]} -> {output_or[i]} -> {output_final[i]} (Очікуваний: {expected_output[i]})")

if np.array_equal(output_final, expected_output):
    print("Тест пройдено успішно! Модель працює коректно.")
else:
    print("Тест не пройдено! Модель має помилки.")

# Побудова окремих графіків для Loss та Accuracy
fig, ax = plt.subplots()
ax.plot(range(1, epochs + 1), loss_values, 'r-', label='Loss')
ax.set_xlabel('Епохи')
ax.set_ylabel('Loss', color='r')
ax.set_title('Графік втрат (Loss)')
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(range(1, epochs + 1), accuracy_values, 'b-', label='Accuracy')
ax.set_xlabel('Епохи')
ax.set_ylabel('Accuracy', color='b')
ax.set_title('Графік точності (Accuracy)')
ax.legend()
plt.show()
