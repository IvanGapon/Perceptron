import numpy as np


class Perceptron:
    def __init__(self, n_features: int, learning_rate: float, epochs: int):
        self.lr = learning_rate
        self.epochs = epochs
        self.weight = np.zeros(n_features)
        self.bias: float = 0.0
        self.history_errors: list[int] = []

    def activate(self, z: float) -> int:
        """
        Проверка порога
        :arg z полученный результат
        :return результат проверки порога
        """
        if z >= 0:
            return 1
        return 0

    def predict(self, x: np.ndarray) -> int:
        """
        Предикт от модели, производим некие операции и проверяем порог
        :arg x вектор признаков объекта
        :return предполагаемый результат
        """
        total_sum = 0.0
        for index, element in enumerate(x):
            total_sum += element * self.weight[index]
        total_sum += self.bias

        return self.activate(total_sum)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Подсчет доли правильных ответов
        :arg X все объекты и их признаки
        :param y целевые значения, которые хотим увидеть
        :return: доля успешных ответов
        """
        count = 0
        for coord, target in zip(X, y):
            prediction = self.predict(coord)
            if prediction == target:
                count += 1

        return count / len(X)

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Функция тренировки. После тренировки проверяем какой результат она нам дает.
        :arg X признаки
        :arg y целевые значения, которые хотим увидеть
        :return доля успешных ответов после тренировки
        """
        self.history_errors = []

        for epoch in range(self.epochs):
            errors = 0
            for coord, target in zip(X, y):
                prediction = self.predict(coord)
                error = target - prediction

                self.weight = self.weight + self.lr * error * coord
                self.bias = self.bias + self.lr * error
                if error != 0:
                    errors += 1

            self.history_errors.append(errors)

        return self.score(X, y)
