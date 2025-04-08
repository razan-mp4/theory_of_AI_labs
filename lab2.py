import numpy as np
import matplotlib.pyplot as plt

# Функція активації (порогова)
def step_function(x):
    return 1 if x >= 0 else 0

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

# Обчислення виходів нейрона "АБО"
output_or = np.array([step_function(np.dot(x, w_or) + b_or) for x in inputs])

# Формування нового входу для нейрона "І"
inputs_and = np.column_stack((inputs[:, 0], output_or))

# Обчислення виходів нейрона "І"
output_final = np.array([step_function(np.dot(x, w_and) + b_and) for x in inputs_and])

# Виведення результатів
print("Вхідні дані (x1, x2) -> OR -> AND -> Вихід")
for i in range(len(inputs)):
    print(f"{inputs[i]} -> {output_or[i]} -> {output_final[i]}")

# Очікувані результати
expected_output = np.array([0, 0, 1, 1])

# Перевірка відповідності очікуваним результатам
if np.array_equal(output_final, expected_output):
    print("Тест пройдено успішно! Модель працює коректно.")
else:
    print("Тест не пройдено! Модель має помилки.")

# Побудова графіка
fig, ax = plt.subplots()
ax.set_title("Графічне представлення роботи перцептрона")
ax.set_xlabel("Вхід x1")
ax.set_ylabel("Вхід x2")

# Відображення точок на графіку
for i in range(len(inputs)):
    color = 'green' if expected_output[i] == 1 else 'red'
    marker = 'o' if expected_output[i] == 1 else 'x'
    ax.scatter(inputs[i][0], inputs[i][1], color=color, marker=marker, s=100,
               label=f"({inputs[i][0]}, {inputs[i][1]}) -> {expected_output[i]}")

# Унікальна легенда
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

# Додавання сітки
ax.grid(True)

# Відображення графіка
plt.show()
