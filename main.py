import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
epochs = 50
val_iter = 5
lr = 1e-4

train = np.loadtxt("./archive/mitbih_train.csv", delimiter=",")
test = np.loadtxt("./archive/mitbih_test.csv", delimiter=",")

X_train = torch.tensor(train[:, :-1], dtype=torch.float32)
y_train = torch.tensor(train[:, -1], dtype=torch.long)

X_test = torch.tensor(test[:, :-1], dtype=torch.float32)
y_test = torch.tensor(test[:, -1], dtype=torch.long)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, d_model=187, labels=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, labels),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    train_loader = DataLoader(
        ECGDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        ECGDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        if epoch % val_iter == 0 or epoch == 1:
            val_loss, val_acc = eval_epoch(model, val_loader, criterion)
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f}"
            )


if __name__ == "__main__":
    main()
