import numpy as np
import cv2

def create_pattern_image(filename, size=(10,10), mode='random'):
    """
    Створює 10×10 чорно-біле зображення і зберігає його у файл filename.
    mode може бути 'random', 'checkerboard', 'diagonal' тощо.
    """

    rows, cols = size
    img = np.zeros((rows, cols), dtype=np.uint8)

    if mode == 'random':
        # Випадкові чорні/білі пікселі
        # 0 = чорний, 255 = білий
        for r in range(rows):
            for c in range(cols):
                val = np.random.choice([0, 255])
                img[r, c] = val

    elif mode == 'checkerboard':
        # "Шахівниця"
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    img[r, c] = 255  # білий
                else:
                    img[r, c] = 0    # чорний

    elif mode == 'diagonal':
        # Діагональна лінія
        for r in range(rows):
            for c in range(cols):
                if r == c:
                    img[r, c] = 255
                else:
                    img[r, c] = 0
    else:
        # За замовчуванням - random
        for r in range(rows):
            for c in range(cols):
                val = np.random.choice([0, 255])
                img[r, c] = val

    # Зберігаємо
    cv2.imwrite(filename, img)
    print(f"Створено файл: {filename}")

if __name__ == "__main__":
    create_pattern_image("pattern0.png", mode='checkerboard')
    create_pattern_image("pattern1.png", mode='diagonal')
    create_pattern_image("pattern2.png", mode='random')
