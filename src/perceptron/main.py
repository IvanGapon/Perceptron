import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

from perceptron.model import Perceptron


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.5,
        flip_y=0.0,
    )

    model = Perceptron(n_features=2, learning_rate=args.lr, epochs=args.epochs)
    final_score = model.train(X, y)

    print("\nрезультат обучения ")
    print(f"итоговая точность: {final_score * 100}%")

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(
        model.history_errors,
        marker="o",
        color="blue",
    )
    ax1.set_title("График ошибок по эпохам")
    ax1.set_ylabel("Количество ошибок")
    ax1.set_xlabel("Эпоха")
    ax1.grid(True)

    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=50)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    x_line = np.array([x_min, x_max])
    y_line = -1 * (model.weight[0] * x_line + model.bias) / model.weight[1]
    ax2.plot(x_line, y_line, color="green", label="Разделяющая прямая")

    ax2.set_title("граница решения")
    ax2.set_xlabel("первый призник")
    ax2.set_ylabel("второй признак")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
