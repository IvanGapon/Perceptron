<h1 align="center">Hi there, I'm <a href="https://github.com/IvanGapon" target="_blank">Ivan</a> </h1>
<h3 align="center">Perceptron </h3>

В репозитории реализована простая линейная моделька перцептрон - принимает признаки, умножает на веса, суммирует и пропускает через функцию активации, выдавая класс (например, 0 или 1). Обучается по правилу: если ошибся — подправляем веса

<h3 align="center">Структура проекта</h3>

- **`src/perceptron/`** — исходный код проекта:
  - [`model.py`](https://github.com/IvanGapon/Perceptrone/blob/main/src/perceptron/model.py) — класс `Perceptron` с логикой обучения, прогнозирования и пороговой функцией.
  - [`main.py`](https://github.com/IvanGapon/Perceptrone/blob/main/src/perceptron/main.py) — точка входа с CLI (поддержка синтетических данных и датасета Iris) и визуализацией через `matplotlib`.
- [`tests/`](https://github.com/IvanGapon/Perceptrone/blob/main/tests/test_perceptron.py) — набор тестов (`pytest`) 
- [`.github/workflows/ci.yml`](https://github.com/IvanGapon/Perceptrone/blob/main/.github/workflows/ci.yml) — конфигурация CI для проверки линкеров, тестов и тайпчекеров при пуше в ветку main
- [`pyproject.toml`](https://github.com/IvanGapon/Perceptrone/blob/main/pyproject.toml) — конфигурация зависимостей и инструментов (`uv`, `ruff`, `mypy`).

<h3 align="center">Как запустить</h3>

Программа запускается через менеджер зависимостей `uv` одной командой из терминала:

 **Запуск на:**
   ```bash
   uv run python src/perceptron/main.py --dataset "synthetic"  --samples 200 --lr 0.1 --epochs 50
  ```
По умолчанию параметр `--dataset` всегда `"synthetic" ` , если мы хотим получить НЕ синтезированное распределение, а из датасета ирис то можем указать `--dataset "iris" ` и тогда параметр `--samples` можно не заполнять

Дефолтные значения : 
* `--dataset="synthetic"`
* `--samples=200`
* `--lr=0.1`
* `--epochs=50`
