import numpy as np

class Perceptron:
    def __init__(self, input_dim, init_type='small', loss_type='bce', l2_lambda=0.0):
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda

        # Инициализация весов
        if init_type == 'zero':
            self.w = np.zeros((input_dim, 1))
        elif init_type == 'large':
            self.w = np.random.normal(0, 10, (input_dim, 1))
        else:  # small random (по умолчанию)
            self.w = np.random.randn(input_dim, 1) * 0.01

        self.b = 0.0

        # Для Momentum SGD
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        if self.loss_type == 'hinge':
            return z  # Для Hinge loss возвращаем сырые логиты
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        if self.loss_type == 'hinge':
            y_h = np.where(y_true == 0, -1, 1) #переводим метки 0 -> -1, 1 -> 1
            loss = np.mean(np.maximum(0, 1 - y_h * y_pred))
        else:  # bce
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 регуляризация
        l2_penalty = (self.l2_lambda / 2) * np.sum(self.w ** 2)
        return loss + l2_penalty

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, optimizer='sgd', beta=0.9):
        train_losses = []
        val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            # перемешиваем данные
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices].reshape(-1, 1)

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                # Прямой проход
                y_pred = self.forward(X_batch)

                # Обратный проход (вычисление градиентов)
                if self.loss_type == 'hinge':
                    y_h = np.where(y_batch == 0, -1, 1)
                    dz = np.where(y_h * y_pred < 1, -y_h, 0)
                else:
                    dz = y_pred - y_batch

                dw = np.dot(X_batch.T, dz) / X_batch.shape[0] + self.l2_lambda * self.w
                db = np.sum(dz) / X_batch.shape[0]

                # Обновление весов
                #v — скорость (накопленный градиент).

                #β (beta) — коэффициент, насколько сильно мы помним старую скорость.
                # Обычно берут 0.9.

                #Если β=0, это обычный SGD (нет памяти).

                #Если β близка к 1, скорость затухает медленно, инерция огромна.
                if optimizer == 'momentum':
                    self.v_w = beta * self.v_w + (1 - beta) * dw
                    self.v_b = beta * self.v_b + (1 - beta) * db
                    self.w -= lr * self.v_w
                    self.b -= lr * self.v_b
                else:  # sgd
                    self.w -= lr * dw
                    self.b -= lr * db

            # Сохраняем loss
            train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

        return train_losses, val_losses

    def predict(self, X):
        y_pred = self.forward(X)
        if self.loss_type == 'hinge':
            return (y_pred >= 0).astype(int).flatten() #astype из бул в инт превращает а
            # flatten вытягивает массив в одномерный
        return (y_pred >= 0.5).astype(int).flatten()