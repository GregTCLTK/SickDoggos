import torch
import numpy as np
import torch.nn as nn


def train(D):

    train, test = D[:200], D[200:]  # Die Auswertung der Aufgabe basiert auf diesem Split

    # Hyper-parameter
    n_steps = 2000
    input_size = 13
    output_size = 1
    learning_rate = 0.01

    # Trainings-Daten vorbereiten
    X = train[:, :-1].astype(np.float32)
    y = train[:, -1].astype(np.float32)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)
    feature_means = torch.mean(X_train, dim=0)

    # Test-Daten vorbereiten
    X = test[:, :-1].astype(np.float32)
    y = test[:, -1].astype(np.float32)
    X_test = torch.from_numpy(X)
    y_test = torch.from_numpy(y)

    # Modell definieren
    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            n_neurons = 20
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size)
            )

        def forward(self, x):
            x = x - feature_means
            out = self.layers(x)
            return out

    model = MLP(input_size)

    # loss and optimizer
    # checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy

    # optimizer
    # Dokumentation: https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Alternative zu Adam
    # Dokumentation: https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
    # momentum = 0.9  # Wert zwischen 0. und 1.
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # trainieren des Modells
    for e in range(n_steps):
        # forward pass
        outputs = model.forward(X_train)[:, 0]  # Xw (linear layer)
        loss = criterion(outputs, y_train)  # sigmoid and cross-entropy loss

        # backward pass (automatically computes gradients)
        optimizer.zero_grad()  # reset gradients (torch accumulates them)
        loss.backward()  # computes gradients

        # Optimierungsschritt durchfuehren
        optimizer.step()

        # berechne Trainings-Accuracy
        outputs = model.forward(X_train)[:, 0]
        pred_y = outputs > 0
        is_correct = torch.eq(pred_y, y_train.byte()).float()
        accuracy_train = torch.mean(is_correct).item()

        # berechne Test-Accuracy
        outputs = model.forward(X_test)[:, 0]
        pred_y = outputs > 0
        is_correct = torch.eq(pred_y, y_test.byte()).float()
        accuracy_test = torch.mean(is_correct).item()

    print(f'Epoch {e}, Loss: {loss:.4f}, Acc train: {accuracy_train:.2f},' \
          f'Acc test: {accuracy_test:.2f}')
    return model

D = np.load('./train_data.npy')

optimized_model_best = train(D)

# Testen des Modells
# Laden der Testdaten
D_test = np.load('./test_data.npy')
X_test = torch.from_numpy(D_test[:, :-1].astype(np.float32))
y_test = torch.from_numpy(D_test[:, -1].astype(np.float32))

# Berechnen der Accuracy
outputs = optimized_model_best.forward(X_test)[:, 0]
pred_y = outputs > 0
is_correct = torch.eq(pred_y, y_test.byte()).float()
accuracy_test = torch.mean(is_correct).item()
print(f'Accuracy test: {accuracy_test:.2f}')
