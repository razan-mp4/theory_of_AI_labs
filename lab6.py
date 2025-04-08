# w_new = w_old + η(x − w_old)

import torch
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Cognitron:
    """
    Спрощена реалізація когнітрону у вигляді набору прототипів.
    Кожен прототип має розмір 28*28 (для зображень MNIST).
    """
    def __init__(self, num_prototypes=10, lr=0.01):
        """
        num_prototypes: кількість прототипів (зазвичай 10, якщо очікуємо 10 різних цифр).
        lr: коефіцієнт навчання (learning rate) для правила Хебба.
        """
        self.num_prototypes = num_prototypes
        self.lr = lr
        
        # Ініціалізуємо прототипи випадковими числами (вектор 784 для кожного).
        self.prototypes = torch.rand(num_prototypes, 28*28)
    
    def train_epoch(self, train_loader):
        """
        Один епох (прогін по всіх батчах) некерованого навчання:
          - Для кожного зображення шукаємо прототип з найбільшим скалярним добутком
          - Оновлюємо тільки "переможця" за геббівським правилом
        """
        for images, _ in train_loader:
            # images shape = (batch_size, 1, 28, 28)
            # Перетворимо зображення на вектори (batch_size, 784)
            images = images.view(-1, 28*28)
            
            for i in range(images.size(0)):
                x = images[i]  # (784,)
                
                # Скалярний добуток (прототипи shape = (num_prototypes, 784))
                # dots -> (num_prototypes,)
                dots = (self.prototypes * x).sum(dim=1)
                
                # Індекс прототипу, що має максимум
                winner_idx = torch.argmax(dots).item()
                
                # Оновлення ваг переможця за правилом:
                # w <- w + alpha * (x - w)
                self.prototypes[winner_idx] += self.lr * (x - self.prototypes[winner_idx])
    
    def predict(self, images):
        """
        Для кожного зображення повертає індекс переможця (прототипа).
        images: тензор (batch_size, 1, 28, 28)
        Повертає: тензор (batch_size,) з індексами прототипів.
        """
        # Перетворюємо на (batch_size, 784)
        images = images.view(-1, 28*28)
        
        # Обчислюємо (batch_size, num_prototypes)
        dots = torch.matmul(images, self.prototypes.t())
        
        # Найбільший скалярний добуток -> переможець
        winners = torch.argmax(dots, dim=1)
        return winners

def label_prototypes(model, loader):
    """
    "Розмітка" прототипів після некерованого навчання:
    1. Для кожного зображення знаходиться переможець,
    2. Запам'ятовуємо справжню цифру (label) цього зразка,
    3. Для кожного прототипа беремо мажоритарну (найчастішу) цифру як ярлик.
    Якщо прототип нічого не "виграв", ставимо -1.
    
    Повертає список довжиною num_prototypes, де кожен елемент - ярлик прототипа.
    """
    num_prototypes = model.prototypes.shape[0]
    assignments = [[] for _ in range(num_prototypes)]
    
    for images, labels in loader:
        winners = model.predict(images)
        for i, w_idx in enumerate(winners):
            assignments[w_idx.item()].append(labels[i].item())
    
    prototype_labels = []
    for i in range(num_prototypes):
        if len(assignments[i]) == 0:
            prototype_labels.append(-1)
        else:
            count_dict = {}
            for lab in assignments[i]:
                count_dict[lab] = count_dict.get(lab, 0) + 1
            # Цифра, що найчастіше траплялася у "перемогах" цього прототипа
            best_label = max(count_dict, key=count_dict.get)
            prototype_labels.append(best_label)
    
    return prototype_labels

def evaluate(model, loader, prototype_labels):
    """
    Обчислення точності:
    1. Для кожного зображення визначаємо переможця (predict),
    2. Бачимо, який ярлик має прототип-переможець,
    3. Порівнюємо із реальним класом зображення.
    """
    correct = 0
    total = 0
    
    for images, labels in loader:
        winners = model.predict(images)
        
        # Перетворюємо кожен індекс прототипа на його ярлик
        mapped = [prototype_labels[w.item()] for w in winners]
        
        for i, lab in enumerate(labels):
            if mapped[i] == lab.item():
                correct += 1
        total += len(labels)
    
    if total == 0:
        return 0.0
    return 100.0 * correct / total

def main():
    # -------------------------
    # 1. Дані MNIST
    # -------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(
        root='.',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='.',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # -------------------------
    # 2. Ініціалізація Cognitron
    # -------------------------
    num_epochs = 5
    cognitron = Cognitron(num_prototypes=10, lr=0.01)
    
    # Масиви для збереження історії точностей
    train_accuracies = []
    test_accuracies = []
    
    # -------------------------
    # 3. Навчання + відслідковування точності
    # -------------------------
    for epoch in range(num_epochs):
        # Виконуємо один епох
        cognitron.train_epoch(train_loader)
        
        # Після епохи – "розмітка" прототипів
        proto_labels = label_prototypes(cognitron, train_loader)
        
        # Обчислення точності на train і test
        train_acc = evaluate(cognitron, train_loader, proto_labels)
        test_acc  = evaluate(cognitron, test_loader, proto_labels)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}%")
    
    # -------------------------
    # 4. Графік зміни точності
    # -------------------------
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Зміна точності (Cogitron) за епохами")
    plt.legend()
    plt.show()
    
    # -------------------------
    # 5. Фінальна розмітка, вивід точності та візуалізація прототипів
    # -------------------------
    proto_labels = label_prototypes(cognitron, train_loader)
    final_train_acc = evaluate(cognitron, train_loader, proto_labels)
    final_test_acc  = evaluate(cognitron, test_loader, proto_labels)
    
    print(f"Фінальна точність на Train: {final_train_acc:.2f}%")
    print(f"Фінальна точність на Test:  {final_test_acc:.2f}%")
    print("Прототипи отримали ярлики:", proto_labels)
    
    # -------------------------
    # 6. Візуалізація кожного прототипу (окремий графік)
    # -------------------------
    for i in range(cognitron.num_prototypes):
        plt.figure()
        proto_img = cognitron.prototypes[i].view(28, 28).detach().numpy()
        plt.imshow(proto_img, cmap='gray')
        plt.title(f"Прототип {i}, присвоєний клас: {proto_labels[i]}")
        plt.show()

if __name__ == "__main__":
    main()
