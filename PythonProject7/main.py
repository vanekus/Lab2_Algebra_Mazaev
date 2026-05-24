import numpy as np
import dataset
from perceptron import Perceptron
import metrics
import visualization as vis


def print_separator(title):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")


def main():
    # ==========================================
    # 1 & 2. Подготовка данных и Обучение (Обязательная часть)
    # ==========================================
    print_separator("ОБЯЗАТЕЛЬНАЯ ЧАСТЬ: БАЗОВОЕ ОБУЧЕНИЕ")
    X, y = dataset.get_mandatory_data()
    X_train, X_test, y_train, y_test = dataset.preprocess_data(X, y)

    model = Perceptron(input_dim=2)
    train_loss, val_loss = model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    # Метрики
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"Accuracy (Train): {metrics.accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Accuracy (Test): {metrics.accuracy_score(y_test, y_test_pred):.4f}")

    # Визуализация
    vis.plot_loss(train_loss, val_loss, "Loss: Базовая модель (lr=0.1, batch=32)")
    vis.plot_decision_boundary(model, X_test, y_test, "Decision Boundary: Базовая модель")

    input("Нажмите Enter для перехода к экспериментам со скоростью обучения (Learning Rate)...")

    # ==========================================
    # 4. Эксперименты: Learning Rate
    # ==========================================
    print_separator("ЭКСПЕРИМЕНТ: СКОРОСТЬ ОБУЧЕНИЯ (Learning Rate)")
    lrs = [0.001, 0.01, 0.5, 1.0]
    for lr in lrs:
        m = Perceptron(input_dim=2)
        t_l, v_l = m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=lr, batch_size=32)
        acc = metrics.accuracy_score(y_test, m.predict(X_test))
        print(f"LR: {lr} | Test Accuracy: {acc:.4f}")
        vis.plot_loss(t_l, v_l, f"Loss curve (LR = {lr})")

    input("Нажмите Enter для перехода к экспериментам с размером батча (Batch Size)...")

    # ==========================================
    # 4. Эксперименты: Batch Size
    # ==========================================
    print_separator("ЭКСПЕРИМЕНТ: РАЗМЕР БАТЧА (Batch Size)")
    batches = [1, 16, 64, 256]
    for b in batches:
        m = Perceptron(input_dim=2)
        t_l, v_l = m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=b)
        acc = metrics.accuracy_score(y_test, m.predict(X_test))
        print(f"Batch Size: {b} | Test Accuracy: {acc:.4f}")
        vis.plot_loss(t_l, v_l, f"Loss curve (Batch Size = {b})")

    input("Нажмите Enter для перехода к экспериментам с инициализацией весов...")

    # ==========================================
    # 4. Эксперименты: Инициализация весов
    # ==========================================
    print_separator("ЭКСПЕРИМЕНТ: ИНИЦИАЛИЗАЦИЯ ВЕСОВ")
    inits = ['zero', 'small', 'large']
    for init in inits:
        m = Perceptron(input_dim=2, init_type=init)
        t_l, v_l = m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        acc = metrics.accuracy_score(y_test, m.predict(X_test))
        print(f"Init: {init.upper()} | Test Accuracy: {acc:.4f}")
        vis.plot_loss(t_l, v_l, f"Loss curve (Init = {init})")
        #если веса огромные, z огромно, ŷ почти 0 или 1. Модель «заперта»:
        # ŷ почти не меняется при небольших изменениях весов,
        # а большие изменения за один шаг невозможны из-за ограниченного lr.
        # В итоге loss застревает, модель не может выбраться из насыщенного состояния.
        # Обучение стоит на месте или идёт крайне медленно.

    print("""
    
    ВЫВОД ПО ИНИЦИАЛИЗАЦИИ: 
    - zero: для перцептрона без скрытых слоев работает (т.к. градиенты все равно обновят веса пропорционально входам), но в глубоких сетях вызовет симметрию.
    - large: работает очень плохо, так как большие значения приводят к насыщению сигмоиды на старте (y^ стремится к 1 или 0), из-за чего градиент равен почти нулю (затухание градиентов) и модель не обучается.
    """)

    input("Нажмите Enter для перехода к бонусным заданиям...")

    # ==========================================
    # ЗАДАНИЯ НА ДОПОЛНИТЕЛЬНЫЕ БАЛЛЫ
    # ==========================================
    print_separator("ДОП. ЗАДАНИЕ 1: Линейно-разделимые данные и XOR")
    X_xor, y_xor = dataset.generate_xor()
    m_xor = Perceptron(input_dim=2)
    m_xor.fit(X_xor, y_xor, X_xor, y_xor, epochs=100)  # для простоты обучаем на всем
    vis.plot_decision_boundary(m_xor, X_xor, y_xor, "Perceptron fails on XOR (Non-linear)")

    print_separator("ДОП. ЗАДАНИЕ 2: Hinge Loss vs BCE")
    m_hinge = Perceptron(input_dim=2, loss_type='hinge')
    t_h, v_h = m_hinge.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.01)
    print(f"Hinge Loss Accuracy: {metrics.accuracy_score(y_test, m_hinge.predict(X_test)):.4f}")
    vis.plot_loss(t_h, v_h, "Loss: Hinge Loss")

    print_separator("ДОП. ЗАДАНИЕ 2: L2-регуляризация")
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    for lam in lambdas:
        m = Perceptron(input_dim=2, l2_lambda=lam)
        t_l, v_l = m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
        acc = metrics.accuracy_score(y_test, m.predict(X_test))
        print(f"λ: {lam} | Test Accuracy: {acc:.4f} | w: {m.w.flatten()}")
        vis.plot_loss(t_l, v_l, f"Loss curve (λ = {lam})")

    print_separator("ДОП. ЗАДАНИЕ 3: Дополнительные метрики и ROC-AUC")
    y_test_pred = model.predict(X_test)
    y_probs = model.forward(X_test).flatten()

    print(f"Precision: {metrics.precision_score(y_test, y_test_pred):.4f}")
    print(f"Recall: {metrics.recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-score: {metrics.f1_score(y_test, y_test_pred):.4f}")

    auc, fpr, tpr = metrics.roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC: {auc:.4f}")
    vis.plot_roc_curve(fpr, tpr, auc)
    vis.plot_errors(model, X_test, y_test, "Ошибки классификации на тестовой выборке")

    print_separator("ДОП. ЗАДАНИЕ 4: SGD Momentum")
    betas = [0.5, 0.9, 0.99]
    for beta in betas:
        m_mom = Perceptron(input_dim=2)
        t_l, v_l = m_mom.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.01, optimizer='momentum', beta=beta)
        print(f"Momentum SGD (beta={beta}) Accuracy: {metrics.accuracy_score(y_test, m_mom.predict(X_test)):.4f}")
        vis.plot_loss(t_l, v_l, f"Loss Momentum SGD (beta = {beta})")


if __name__ == "__main__":
    main()
