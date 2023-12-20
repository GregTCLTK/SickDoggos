import torch
import numpy as np
import torch.nn as nn


def train(D):
    # Teilt das Dataset D in Trainings- und Testdaten auf. Die ersten 200 Einträge sind Trainingsdaten, der Rest sind Testdaten.
    train, test = D[:200], D[200:]

    # Definiert die Hyperparameter für das Training.
    # n_steps: Anzahl der Trainingsschritte
    # input_size: Anzahl der Eingangsmerkmale
    # output_size: Anzahl der Ausgangsmerkmale
    # learning_rate: Lernrate für den Optimierer
    n_steps = 2000
    input_size = 13
    output_size = 1
    learning_rate = 0.01

    # Bereitet die Trainingsdaten vor.
    # X: Eingangsmerkmale
    # y: Zielmerkmale
    # X_train und y_train sind Tensoren, die aus den Numpy-Arrays X und y erstellt wurden.
    X = train[:, :-1].astype(np.float32)
    y = train[:, -1].astype(np.float32)
    X_train = torch.from_numpy(X)
    y_train = torch.from_numpy(y)
    # Berechnet den Durchschnitt der Merkmale für die spätere Normalisierung.
    feature_means = torch.mean(X_train, dim=0)

    # Bereitet die Testdaten vor.
    # X_test und y_test sind Tensoren, die aus den Numpy-Arrays X und y erstellt wurden.
    X = test[:, :-1].astype(np.float32)
    y = test[:, -1].astype(np.float32)
    X_test = torch.from_numpy(X)
    y_test = torch.from_numpy(y)

    # Definiert das Modell als eine Klasse, die von nn.Module erbt.
    # Das Modell ist ein mehrschichtiges Perzeptron (MLP).
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

    # Definiert das Modell. Hier wird die zuvor definierte MLP-Klasse verwendet.
    # Die Anzahl der Eingangsmerkmale wird als Parameter übergeben.
    model = MLP(input_size)

    # Definiert die Verlustfunktion. In diesem Fall wird die BCEWithLogitsLoss-Funktion verwendet,
    # die eine Kombination aus einer Sigmoid-Aktivierung und einer binären Kreuzentropie-Verlustfunktion ist.
    criterion = nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy

    # Definiert den Optimierer. Hier wird der Adam-Optimierer verwendet.
    # Die Parameter des Modells und die Lernrate werden als Parameter übergeben.
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

    print('Model trainiert')
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
