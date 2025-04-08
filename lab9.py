import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

def build_car_speed_controller():
    """
    Створює приклад нечіткої системи для керування швидкістю/прискоренням автомобіля.
    Вхід: distance (0..150 м), speed (0..120 км/год).
    Вихід: acceleration (-10..10), де від’ємне – гальмування, додатне – прискорення.
    """

    # 1) Нечіткі змінні (Antecedent: distance, speed; Consequent: acceleration)
    distance = ctrl.Antecedent(np.arange(0, 151, 1), 'distance')       # 0..150 м
    speed    = ctrl.Antecedent(np.arange(0, 121, 1), 'speed')          # 0..120 км/год
    acceleration = ctrl.Consequent(np.arange(-10, 11, 1), 'acceleration')  # -10..+10

    # 2) Функції належності
    # Distance:
    distance['very_close'] = fuzz.trimf(distance.universe, [0, 0, 30])
    distance['close']      = fuzz.trimf(distance.universe, [0, 30, 60])
    distance['medium']     = fuzz.trimf(distance.universe, [40, 75, 110])
    distance['far']        = fuzz.trimf(distance.universe, [90, 150, 150])

    # Speed:
    speed['low']    = fuzz.trimf(speed.universe, [0, 0, 50])
    speed['medium'] = fuzz.trimf(speed.universe, [30, 60, 90])
    speed['high']   = fuzz.trimf(speed.universe, [80, 120, 120])

    # Acceleration:
    acceleration['strong_brake']   = fuzz.trimf(acceleration.universe, [-10, -10, -4])
    acceleration['light_brake']    = fuzz.trimf(acceleration.universe, [-8, -3, 0])
    acceleration['keep']           = fuzz.trimf(acceleration.universe, [-1, 0, 1])
    acceleration['light_accel']    = fuzz.trimf(acceleration.universe, [0, 3, 7])
    acceleration['strong_accel']   = fuzz.trimf(acceleration.universe, [5, 10, 10])

    # 3) База правил (IF–THEN)
    rule1 = ctrl.Rule(
        antecedent=(distance['very_close'] | distance['close']),
        consequent=acceleration['strong_brake']
    )
    rule2 = ctrl.Rule(
        antecedent=(distance['far'] & speed['low']),
        consequent=acceleration['strong_accel']
    )
    rule3 = ctrl.Rule(
        antecedent=(distance['medium'] & speed['medium']),
        consequent=acceleration['keep']
    )
    rule4 = ctrl.Rule(
        antecedent=(distance['far'] & speed['high']),
        consequent=acceleration['keep']
    )
    rule5 = ctrl.Rule(
        antecedent=(distance['close'] & speed['high']),
        consequent=acceleration['light_brake']
    )

    car_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    car_sim  = ctrl.ControlSystemSimulation(car_ctrl)

    # Повертаємо і Antecedent/Consequent об'єкти, і симулятор
    # щоб у main() ми могли їх також відмалювати
    return distance, speed, acceleration, car_sim

def plot_fuzzy_variable(variable, title=None):
    """
    Побудова кривих функцій належності (membership) для кожного терму змінної variable.
    variable: Antecedent або Consequent (з полем .universe і .terms)
    """
    x = variable.universe
    plt.figure()
    for term_name, membership_func in variable.terms.items():
        # membership_func - це fuzzymf.MembershipFunction
        # щоб отримати масив значень, використаємо fuzz.interp_membership
        # (це простий спосіб відмалювати криву)
        y = fuzz.interp_membership(x, membership_func.mf, x)
        plt.plot(x, y, label=term_name)

    plt.title(title if title else variable.label)
    plt.xlabel(variable.label)
    plt.ylabel('Degree of membership')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    # Побудова системи
    distance, speed, acceleration, car_sim = build_car_speed_controller()

    # 1) Відмалюємо функції належності (до обчислень)
    plot_fuzzy_variable(distance, "Distance membership")
    plot_fuzzy_variable(speed, "Speed membership")
    plot_fuzzy_variable(acceleration, "Acceleration membership")

    # Тестовий сценарій: distance=45 м, speed=90 км/год
    dist_input = 45
    speed_input = 90

    car_sim.input['distance'] = dist_input
    car_sim.input['speed']    = speed_input

    car_sim.compute()
    accel_output = car_sim.output['acceleration']

    print("=== Fuzzy Car Speed Controller ===")
    print(f"Distance to car in front: {dist_input} м")
    print(f"Speed of our car: {speed_input} км/год")
    print(f"Suggested acceleration = {accel_output:.2f}")


if __name__ == "__main__":
    main()
