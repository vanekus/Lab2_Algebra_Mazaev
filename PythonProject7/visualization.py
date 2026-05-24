import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train_losses, val_losses, title="Loss curve"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#Эта функция позволяет наглядно увидеть, как модель разделила пространство:
# красная зона — предсказания класса 1, синяя — класса 0, точки — реальные данные.
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    plt.figure(figsize=(8, 6))

    # Создаем сетку
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')

    # Рисуем разделяющую прямую w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1 + b)/w2
    w1, w2 = model.w[0, 0], model.w[1, 0]
    b = model.b
    if w2 != 0:
        x_line = np.array([x_min, x_max])
        y_line = -(w1 * x_line + b) / w2
        plt.plot(x_line, y_line, 'k-', linewidth=2, label='w^Tx + b = 0')
        plt.ylim(y_min, y_max)

    plt.title(title)
    plt.legend()
    plt.show()

#Рисует ROC-кривую — график зависимости True Positive Rate (TPR) от False Positive Rate (FPR)
# при изменении порога.
#Также рисует диагональную пунктирную линию — это «случайный классификатор» (AUC = 0.5).
#В легенде выводит значение AUC (площадь под кривой), переданное третьим аргументом.
def plot_roc_curve(fpr, tpr, auc, title="ROC Curve"):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

#Показывает на плоскости все точки тестовых данных, но по-особому:

#Правильно классифицированные точки (Correct) — обычные цветные кружки (цвет по истинному классу).

#Ошибочно классифицированные (Error) — обведены красным кругом без заливки,
# увеличенного размера (s=100), чтобы их было сразу видно.

#Так можно понять, где модель ошибается и есть ли систематические ошибки.
def plot_errors(model, X, y_true, title="Misclassified points"):
    y_pred = model.predict(X)
    errors = y_true != y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(X[~errors, 0], X[~errors, 1], c=y_true[~errors], cmap='coolwarm', edgecolors='k', label='Correct')
    plt.scatter(X[errors, 0], X[errors, 1], facecolors='none', edgecolors='red', s=100, linewidths=2, label='Error')
    plt.title(title)
    plt.legend()
    plt.show()