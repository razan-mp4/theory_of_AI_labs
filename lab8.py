import math
import random
import matplotlib.pyplot as plt

def euclidean_distance(a, b):
    """Відстань між двома точками a і b у 2D."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def build_distance_matrix(coords):
    """Створює матрицю відстаней для списку координат coords."""
    n = len(coords)
    dist_matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = euclidean_distance(coords[i], coords[j])
    return dist_matrix

def initialize_pheromone(n_cities, initial_pher=1.0):
    """Ініціалізуємо рівень феромону на всіх ребрах."""
    pheromone = [[initial_pher for _ in range(n_cities)] for _ in range(n_cities)]
    return pheromone

def total_distance(dist_matrix, tour):
    """Обчислює сумарну довжину маршруту 'tour'."""
    d = 0.0
    for i in range(len(tour) - 1):
        d += dist_matrix[tour[i]][tour[i+1]]
    # При потребі, можна замкнути тур (додати перехід останній->перший)
    return d

def ant_tour(pheromone, dist_matrix, alpha, beta):
    """
    Побудова маршруту однією "мурахою".
    На вході:
    - pheromone: матриця феромону
    - dist_matrix: матриця відстаней
    - alpha, beta: ваги для феромону та "привабливості" (1/відстань).
    """
    n = len(dist_matrix)
    # список міст (0..n-1)
    unvisited = list(range(n))
    # випадково виберемо старт
    start = random.choice(unvisited)
    tour = [start]
    unvisited.remove(start)
    
    # поки є невідвідані міста
    current_city = start
    while unvisited:
        # обчислити "привабливість" переходів
        probabilities = []
        pher_row = pheromone[current_city]
        for next_city in unvisited:
            tau = pher_row[next_city]**alpha
            eta = (1.0 / dist_matrix[current_city][next_city])**beta if dist_matrix[current_city][next_city] > 0 else 0
            probabilities.append(tau * eta)
        # нормуємо й вибираємо наступне місто за імовірнісним правилом
        s = sum(probabilities)
        if s == 0:
            # Якщо всі ймовірності нулі, виберемо випадково
            chosen = random.choice(unvisited)
        else:
            r = random.random() * s
            cum_sum = 0.0
            for idx, p in enumerate(probabilities):
                cum_sum += p
                if cum_sum >= r:
                    chosen = unvisited[idx]
                    break
        tour.append(chosen)
        unvisited.remove(chosen)
        current_city = chosen
    
    return tour


def update_pheromone(pheromone, tours, dist_matrix, rho=0.5, Q=100.0):
    """
    Оновлюємо феромон (випаровування + додавання).
    - rho: коеф. випаровування
    - Q: "кількість" феромону
    """
    n = len(pheromone)
    # 1) випаровування
    for i in range(n):
        for j in range(n):
            pheromone[i][j] *= (1 - rho)
    
    # 2) додавання феромону за пройденими шляхами мурах
    for tour in tours:
        length = total_distance(dist_matrix, tour)
        deposit = Q / length if length > 0 else 0
        for k in range(len(tour) - 1):
            i, j = tour[k], tour[k+1]
            pheromone[i][j] += deposit
            pheromone[j][i] += deposit  # двонаправлено

def ant_colony_optimization(coords, n_ants=10, n_iter=50, alpha=1, beta=2, rho=0.5, Q=100):
    """
    Основна функція ACO.
    - coords: список координат міст
    - n_ants: скільки "мурах" запускаємо на ітерацію
    - n_iter: скільки ітерацій
    - alpha, beta: ваги феромону і привабливості
    - rho: випаровування
    - Q: кількість феромону для оновлення
    """
    dist_matrix = build_distance_matrix(coords)
    n_cities = len(coords)
    # Ініціалізуємо феромон
    pheromone = initialize_pheromone(n_cities, initial_pher=1.0)
    
    best_tour = None
    best_length = float('inf')
    
    lengths_history = []
    
    for iteration in range(n_iter):
        # Збір маршрутів від усіх мурах
        tours = []
        for _ in range(n_ants):
            tour = ant_tour(pheromone, dist_matrix, alpha, beta)
            tours.append(tour)
        
        # Оновлення феромону
        update_pheromone(pheromone, tours, dist_matrix, rho, Q)
        
        # Перевіряємо, чи не знайшли ми кращий тур
        best_in_iter = None
        best_in_iter_length = float('inf')
        for t in tours:
            l = total_distance(dist_matrix, t)
            if l < best_in_iter_length:
                best_in_iter_length = l
                best_in_iter = t
        
        if best_in_iter_length < best_length:
            best_length = best_in_iter_length
            best_tour = best_in_iter
        
        lengths_history.append(best_length)
        print(f"Ітерація {iteration+1}/{n_iter}, найкращий тур у цій ітерації = {best_in_iter_length:.2f}, глобальний = {best_length:.2f}")
    
    return best_tour, best_length, lengths_history

def main():
    # 1) Генеруємо випадкові координати (можна замінити на реальні)
    random.seed(0)
    n_cities = 10
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_cities)]
    
    # 2) Запускаємо ACO
    best_tour, best_length, lengths_history = ant_colony_optimization(
        coords, 
        n_ants=10, 
        n_iter=50, 
        alpha=1, 
        beta=2, 
        rho=0.5, 
        Q=100
    )
    
    print("Найкращий знайдений маршрут:", best_tour)
    print("Його довжина:", best_length)
    
    # 3) Будуємо графік зміни кращої довжини маршруту
    plt.plot(lengths_history)
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраща довжина")
    plt.title("ACO: зміна довжини маршруту з ітераціями")
    plt.show()

if __name__ == "__main__":
    main()
