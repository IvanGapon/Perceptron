import numpy as np

from perceptron.model import Perceptron


def test_activation_function() -> None:
    """
    проверка пороговой функции
    """
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=1)

    assert model.activate(0.0) == 1
    assert model.activate(1.4) == 1
    assert model.activate(1000.0) == 1

    assert model.activate(-0.8) == 0
    assert model.activate(-6.0) == 0
    assert model.activate(-10000.0) == 0


def test_predict() -> None:
    """
    проверка обучаемости и корректности обученности модели
    """
    X = np.array(
        [
            [1.0, 1.0],
            [1.5, 2.0],
            [2.0, 1.5],
            [1.2, 2.5],
            [2.2, 2.2],
            [5.0, 5.0],
            [5.5, 6.0],
            [6.0, 5.5],
            [5.2, 6.5],
            [6.2, 6.2],
        ]
    )
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    model = Perceptron(n_features=2, learning_rate=0.1, epochs=50)
    score = model.train(X, y)

    assert score == 1.0
    assert model.predict(np.array([1.5, 1.5])) == 1
    assert model.predict(np.array([5.5, 5.5])) == 0


def test_score() -> None:
    """
    Проверка метода подсчета точности
    """
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=1)
    model.weight = np.array([1.0, 1.0])
    model.bias = -1.5

    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1])

    accuracy = model.score(X, y)
    assert accuracy == 1.0


def test_history_errors() -> None:
    """
    проверяем историю оишбок и сами ошибки
    """
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1])

    epochs = 5
    model = Perceptron(n_features=2, learning_rate=0.1, epochs=epochs)
    model.train(X, y)

    assert len(model.history_errors) == epochs
    assert all(isinstance(err, int) and err >= 0 for err in model.history_errors)


def test_parameters() -> None:
    """
    Корректность аргументов
    """
    model = Perceptron(n_features=3, learning_rate=0.25, epochs=15)

    assert model.lr == 0.25
    assert model.epochs == 15
    assert len(model.weight) == 3
    assert np.all(model.weight == 0.0)
    assert model.bias == 0.0
    assert model.history_errors == []
