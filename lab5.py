# Хебба (Hebb rule)
# wᵢⱼ = (1/p) ∑ (xᵢ^μ xⱼ^μ), де μ = 1, 2, …, p; i ≠ j

# оновлення стану нейронів
# sᵢ ← sign(∑ wᵢⱼ sⱼ).

# мінімізація функції енергії
# E(s) = -(1/2) ∑∑ wᵢⱼ sᵢ sⱼ.

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, n):
        """
        n: загальна кількість пікселів (нейронів), наприклад 10×10 = 100
        """
        self.n = n
        self.W = np.zeros((n, n), dtype=np.float32)  # матриця ваг

    def train(self, patterns):
        """
        patterns: список векторів форми (n,), значення +1 або -1
        Наприклад, список довжиною 5 патернів, кожен з яких 100-розмірний
        """
        # Скидаємо W на нуль
        self.W = np.zeros((self.n, self.n), dtype=np.float32)

        # Правило Хопфілда (Hebb rule)
        for p in patterns:
            p = p.reshape(-1, 1)  # Робимо стовпець
            self.W += p @ p.T  # outer product

        # Занулюємо діагональ
        np.fill_diagonal(self.W, 0)

        # Нормуємо на кількість патернів
        self.W /= len(patterns)

    def energy(self, state):
        """
        Обчислюємо «енергію» Хопфілда (має зменшуватися при переході до атрактора).
        state: вектор +1/-1 форми (n,)
        E = -1/2 * sum_i(sum_j( w_ij * s_i * s_j ))
        """
        return -0.5 * state @ (self.W @ state)

    def predict_async(self, state, max_iter=100):
        """
        Асинхронне оновлення (по одному нейрону в довільному порядку).
        state: копія вектора (n,)
        max_iter: максимальна кількість проходів
        Повертає (final_state, history), де history – список енергій на кожній ітерації.
        """
        s = np.copy(state)
        energy_history = []

        for _ in range(max_iter):
            # Збираємо енергію на початку кожного "циклу"
            current_energy = self.energy(s)
            energy_history.append(current_energy)

            # Якщо в процесі бажаємо моніторити, чи змінилося щось
            changed = False

            # Перемішуємо індекси, щоб не було завжди одного порядку оновлення
            indices = np.arange(self.n)
            np.random.shuffle(indices)

            for i in indices:
                net_input = self.W[i, :] @ s
                new_state = 1 if net_input >= 0 else -1
                if new_state != s[i]:
                    changed = True
                s[i] = new_state

            # Якщо за один повний прохід не змінився жоден нейрон - система стабілізувалася
            if not changed:
                break

        # Додаємо енергію підсумкового стану
        energy_history.append(self.energy(s))
        return s, energy_history

    def predict_sync(self, state, max_iter=100):
        """
        Синхронне оновлення (усі нейрони одночасно).
        Аналогічно, повертає (final_state, history).
        """
        s = np.copy(state)
        energy_history = []

        for _ in range(max_iter):
            current_energy = self.energy(s)
            energy_history.append(current_energy)

            new_s = np.copy(s)
            net_input = self.W @ s
            new_s = np.where(net_input >= 0, 1, -1)

            # Перевірка змін
            if np.array_equal(new_s, s):
                # Стан не змінився => стабілізація
                break

            s = new_s

        # Додаємо останню енергію
        energy_history.append(self.energy(s))
        return s, energy_history

# --- Допоміжні функції ---

def load_and_binarize_image(filepath, size=(10,10), threshold=128):
    """
    Завантажуємо картинку через OpenCV,
    переводимо в градації сірого, масштаб, бінаризуємо.
    Повертає вектор форми (size[0]*size[1],) з +1/–1.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не знайдено.")

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    # Бінаризація
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Перетворюємо 0/255 у -1/+1
    # Якщо біле=255 => +1, чорне=0 => -1
    vector = []
    for pixel in bin_img.flatten():
        if pixel > 0:  # 255 => білий
            vector.append(+1)
        else:         # 0 => чорний
            vector.append(-1)

    return np.array(vector, dtype=np.int8)

def add_noise_to_pattern(pattern, noise_level=0.1):
    """
    Випадковим чином інвертуємо частку (noise_level) бітів у патерні.
    noise_level: від 0 до 1. Наприклад, 0.1 => 10% пікселів інвертуються.
    """
    noisy = np.copy(pattern)
    n = len(pattern)
    num_flips = int(n * noise_level)

    flip_indices = random.sample(range(n), num_flips)
    for idx in flip_indices:
        noisy[idx] = -noisy[idx]

    return noisy

def vector_to_image(vec, size=(10,10)):
    """
    Перетворює вектор +1/-1 назад у 2D-масив 0..255 для візуалізації/збереження.
    """
    arr = np.where(vec > 0, 255, 0).reshape(size)
    return arr.astype(np.uint8)

# --- Демонстраційний main-скрипт ---

if __name__ == "__main__":
    # Приклад шляху до файлів (потрібно, щоб pattern0.png і т.д. існували)
    img_files = ["pattern0.png", "pattern1.png", "pattern2.png"]

    # Завантажуємо патерни і зберігаємо у список
    patterns = []
    for f in img_files:
        p = load_and_binarize_image(f, size=(10,10))
        patterns.append(p)

    # Ініціалізуємо мережу Хопфілда на 10x10=100 нейронів
    hop_net = HopfieldNetwork(n=100)

    # Навчання на наших патернах
    hop_net.train(patterns)

    # Беремо якусь із заучених картинок, робимо шум (30%)
    test_index = 0
    original = patterns[test_index]
    noisy_pattern = add_noise_to_pattern(original, noise_level=0.3)

    print("ОРИГІНАЛ:", img_files[test_index])
    print("ШУМОВИЙ ПАТЕРН: 30% flipped")

    # Перевірка з асинхронним оновленням
    recovered_async, energy_hist_async = hop_net.predict_async(noisy_pattern, max_iter=50)

    # Перевірка з синхронним оновленням
    recovered_sync, energy_hist_sync = hop_net.predict_sync(noisy_pattern, max_iter=50)

    # Виводимо інформацію
    print("Кінцева енергія (async):", energy_hist_async[-1])
    print("Кінцева енергія (sync):", energy_hist_sync[-1])

    # Збережемо результати для наочності
    cv2.imwrite("original.png", vector_to_image(original, (10,10)))
    cv2.imwrite("noisy.png", vector_to_image(noisy_pattern, (10,10)))
    cv2.imwrite("recovered_async.png", vector_to_image(recovered_async, (10,10)))
    cv2.imwrite("recovered_sync.png", vector_to_image(recovered_sync, (10,10)))

    print("Збережено файли: original.png, noisy.png, recovered_async.png, recovered_sync.png")

    print("\nПрогрес енергії (async):", energy_hist_async)
    print("Прогрес енергії (sync):", energy_hist_sync)

    # --- ДОДАТИ ГРАФІКИ ЕНЕРГІЇ ---
    plt.plot(energy_hist_async, label='Energy (Async)')
    plt.plot(energy_hist_sync, label='Energy (Sync)')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Hopfield Network Energy Convergence')
    plt.legend()
    plt.show()
