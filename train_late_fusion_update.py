import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# Set seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# =============================
# 1. Load Data
# =============================
data = np.load("data_pamap2_federated_top8.npz")
X = data["X"]  # shape: (N, 100, 52)
y = data["y"]
client_ids = data["client_ids"]

# Split feature by modality
X_hand = X[:, :, 0:17]    # col 4-20
X_chest = X[:, :, 17:34]  # col 21-37
X_ankle = X[:, :, 34:51]  # col 38-54

# =============================
# 2. Dataset and Dataloader
# =============================
class PAMAPLateFusionDataset(Dataset):
    def __init__(self, hand, chest, ankle, labels):
        self.hand = torch.tensor(hand, dtype=torch.float32)
        self.chest = torch.tensor(chest, dtype=torch.float32)
        self.ankle = torch.tensor(ankle, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.hand[idx], self.chest[idx], self.ankle[idx], self.labels[idx]

# Group data by client
client_data = defaultdict(lambda: {"hand": [], "chest": [], "ankle": [], "y": []})
for i in range(len(y)):
    cid = client_ids[i]
    client_data[cid]["hand"].append(X_hand[i])
    client_data[cid]["chest"].append(X_chest[i])
    client_data[cid]["ankle"].append(X_ankle[i])
    client_data[cid]["y"].append(y[i])

# =============================
# 3. Late Fusion Model
# =============================
class LateFusionLSTM(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, num_classes=8):
        super().__init__()
        self.lstm_hand = nn.LSTM(input_dims[0], hidden_dim, batch_first=True)
        self.lstm_chest = nn.LSTM(input_dims[1], hidden_dim, batch_first=True)
        self.lstm_ankle = nn.LSTM(input_dims[2], hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_hand, x_chest, x_ankle):
        _, (h_hand, _) = self.lstm_hand(x_hand)
        _, (h_chest, _) = self.lstm_chest(x_chest)
        _, (h_ankle, _) = self.lstm_ankle(x_ankle)
        h_cat = torch.cat([h_hand[-1], h_chest[-1], h_ankle[-1]], dim=1)
        return self.classifier(h_cat)

# =============================
# 4. Federated Learning Training
# =============================
def train_local(model, dataloader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        total_loss = 0
        for x_hand, x_chest, x_ankle, labels in dataloader:
            x_hand, x_chest, x_ankle, labels = x_hand.to(device), x_chest.to(device), x_ankle.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x_hand, x_chest, x_ankle)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Local epoch {epoch+1}/5, loss: {total_loss / len(dataloader):.4f}")
    return model.state_dict()

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_hand, x_chest, x_ankle, labels in dataloader:
            x_hand, x_chest, x_ankle, labels = x_hand.to(device), x_chest.to(device), x_ankle.to(device), labels.to(device)
            outputs = model(x_hand, x_chest, x_ankle)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# =============================
# 5. Run FL Training
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = len(client_data)
input_dims = [17, 17, 17]

# Create global test set
all_test_hand, all_test_chest, all_test_ankle, all_test_labels = [], [], [], []
for cid in list(client_data.keys()):
    X_train, X_test, y_train, y_test, ch_train, ch_test, an_train, an_test = train_test_split(
        client_data[cid]["hand"],
        client_data[cid]["y"],
        client_data[cid]["chest"],
        client_data[cid]["ankle"],
        test_size=0.2,
        random_state=SEED
    )
    client_data[cid]["hand"] = X_train
    client_data[cid]["chest"] = ch_train
    client_data[cid]["ankle"] = an_train
    client_data[cid]["y"] = y_train

    all_test_hand += list(X_test)
    all_test_chest += list(ch_test)
    all_test_ankle += list(an_test)
    all_test_labels += list(y_test)

test_dataset = PAMAPLateFusionDataset(all_test_hand, all_test_chest, all_test_ankle, all_test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize global model
global_model = LateFusionLSTM(input_dims).to(device)
global_weights = global_model.state_dict()

for rnd in range(1, 21):
    print(f"\n--- Round {rnd} ---")
    local_weights = []

    for cid in sorted(client_data.keys()):
        print(f" Training client {cid}...")
        model = LateFusionLSTM(input_dims).to(device)
        model.load_state_dict(global_weights)
        dataset = PAMAPLateFusionDataset(client_data[cid]["hand"], client_data[cid]["chest"], client_data[cid]["ankle"], client_data[cid]["y"])
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        w = train_local(model, dataloader, device)
        local_weights.append(w)

    # FedAvg
    new_state_dict = {k: torch.stack([w[k] for w in local_weights], dim=0).mean(0) for k in global_weights.keys()}
    global_model.load_state_dict(new_state_dict)
    global_weights = new_state_dict

    acc = evaluate(global_model, test_loader, device)
    print(f"Test accuracy after round {rnd}: {acc * 100:.2f}%")

