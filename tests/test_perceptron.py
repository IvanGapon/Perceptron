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
