import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import defaultdict

# ======== 1. Define LSTM branch for each modal ========
class LSTMBranch(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0)  # dropout only if num_layers > 1

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: [num_layers, B, H]
        return hn[-1]  # take output from the last LSTM layer

# ======== 2. Define Late Fusion Model with 3 branches ========
class LateFusionLSTM(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_classes=8):
        super().__init__()
        self.branches = nn.ModuleList([LSTMBranch(input_size, hidden_size) for _ in range(3)])
        self.classifier = nn.Linear(3 * hidden_size, num_classes)

    def forward(self, x_list):
        features = torch.stack([branch(x) for branch, x in zip(self.branches, x_list)], dim=1)
        concat = features.view(features.size(0), -1)
        return self.classifier(concat)

# ======== 3. Train one client ========
def train_one_client(model, loader, num_epochs, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x1 = x[:, :, 0:17]   # Hand
            x2 = x[:, :, 17:34]  # Chest
            x3 = x[:, :, 34:51]  # Ankle

            optimizer.zero_grad()
            out = model([x1, x2, x3])
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"  Local epoch {epoch+1}/{num_epochs}, loss: {np.mean(losses):.4f}")

    return {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

# ======== 4. Aggregate ========
def average_weights(weight_list):
    avg_weights = defaultdict(float)
    for key in weight_list[0].keys():
        avg_weights[key] = sum(w[key] for w in weight_list) / len(weight_list)
    return avg_weights

# ======== 5. Evaluate ========
def evaluate_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x1 = x[:, :, 0:17]
            x2 = x[:, :, 17:34]
            x3 = x[:, :, 34:51]
            out = model([x1, x2, x3])
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

# ======== 6. Federated Learning ========
def federated_learning(X, y, client_ids, num_rounds=20, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clients = np.unique(client_ids)
    global_model = LateFusionLSTM().to(device)

    # Create global test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.long)),
                             batch_size=64, shuffle=False)

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd} ---")
        local_weights = []

        for cid in clients:
            idx = client_ids == cid
            X_client = X[idx]
            y_client = y[idx]
            loader = DataLoader(TensorDataset(torch.tensor(X_client, dtype=torch.float32),
                                              torch.tensor(y_client, dtype=torch.long)),
                                batch_size=64, shuffle=True)
            print(f" Training client {cid}...")
            local_model = LateFusionLSTM().to(device)
            local_model.load_state_dict(global_model.state_dict())
            weights = train_one_client(local_model, loader, num_epochs, device)
            local_weights.append(weights)

        avg_weights = average_weights(local_weights)
        global_model.load_state_dict(avg_weights)

        acc = evaluate_model(global_model, test_loader, device)
        print(f"Test accuracy after round {rnd}: {acc:.2f}%")

    print("Training finished.")

# ======== 7. Load Data ========
def load_data():
    data = np.load("data_pamap2_federated_top8_filtered.npz")
    return data['X'], data['y'], data['client_ids']

if __name__ == '__main__':
    X, y, client_ids = load_data()
    federated_learning(X, y, client_ids, num_rounds=20, num_epochs=5)

