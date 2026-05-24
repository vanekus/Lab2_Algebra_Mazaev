import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_mandatory_data():
    """Генерация данных для обязательной части."""
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0, n_informative=2,
        random_state=42, n_clusters_per_class=1
    )
    return X, y


def preprocess_data(X, y, test_size=0.3):
    """Разбиение данных и Z-score стандартизация."""
    # Стратифицированное разбиение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Стандартизация (обучаем только на train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# --- Задание на дополнительные баллы 1: Собственный генератор ---
def generate_linear(n_samples=500, noise_prob=0.0): #линейно разделимые данные с возможным шумом
    np.random.seed(42)
    # Два гауссовых облака
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    y1 = np.zeros(n_samples // 2)
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    y2 = np.ones(n_samples // 2)

    X = np.vstack([X1, X2]) #np.vstack складывает их по вертикали
    # в одну матрицу X размера n_samples × 2.
    y = np.concatenate([y1, y2])
    #np.concatenate объединяет метки в один массив.

    # Добавление шума (сдвиг метки)
    if noise_prob > 0:
        flip_mask = np.random.rand(n_samples) < noise_prob
        y[flip_mask] = 1 - y[flip_mask]

    return X, y


def generate_xor(n_samples=500):
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y
