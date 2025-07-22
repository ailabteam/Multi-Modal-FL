import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

# --- Config ---
DATA_PATH = 'data_pamap2_federated_top8_filtered.npz'  # file của bạn
NUM_CLIENTS = 7
BATCH_SIZE = 64
LOCAL_EPOCHS = 5
ROUNDS = 10
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load data ---
data = np.load(DATA_PATH)
X = data['X']  # (N, 100, 52)
y = data['y']
client_ids = data['client_ids']

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, clients shape: {client_ids.shape}")

# --- Modal slices ---
# Hand: features 0-16 (17 features)
# Chest: features 17-33 (17 features)
# Ankle: features 34-50 (17 features)
modal_slices = {
    'hand': slice(0, 17),
    'chest': slice(17, 34),
    'ankle': slice(34, 51),
}

# --- Model ---
class ModalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * 100, 128),  # 100 timesteps
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class LateFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_net = ModalMLP(input_dim=17)
        self.chest_net = ModalMLP(input_dim=17)
        self.ankle_net = ModalMLP(input_dim=17)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 classes
        )
    def forward(self, x):
        hand_x = x[:, :, modal_slices['hand']]
        chest_x = x[:, :, modal_slices['chest']]
        ankle_x = x[:, :, modal_slices['ankle']]

        h = self.hand_net(hand_x)
        c = self.chest_net(chest_x)
        a = self.ankle_net(ankle_x)

        fused = torch.cat([h, c, a], dim=1)
        out = self.classifier(fused)
        return out

# --- Prepare client data ---
clients_data = {}
for client in np.unique(client_ids):
    idxs = np.where(client_ids == client)[0]
    clients_data[client] = (X[idxs], y[idxs])

def get_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_local(model, dataloader, epochs=LOCAL_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        print(f"  Local epoch {epoch+1}/{epochs}, loss: {running_loss / len(dataloader.dataset):.4f}")

def average_weights(w_list):
    avg_w = OrderedDict()
    for k in w_list[0].keys():
        avg_w[k] = torch.stack([w[k].float() for w in w_list], 0).mean(0)
    return avg_w

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total

# --- Prepare test set (10%) ---
np.random.seed(42)
all_indices = np.arange(len(X))
np.random.shuffle(all_indices)
split_idx = int(0.9 * len(all_indices))
train_idx = all_indices[:split_idx]
test_idx = all_indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
test_loader = get_dataloader(X_test, y_test, shuffle=False)

# --- FL training ---
global_model = LateFusionMLP().to(DEVICE)
global_weights = global_model.state_dict()

for r in range(1, ROUNDS+1):
    print(f"\n--- Round {r} ---")
    local_weights = []
    local_sizes = []
    for client in clients_data.keys():
        local_model = LateFusionMLP().to(DEVICE)
        local_model.load_state_dict(global_weights)
        Xc, yc = clients_data[client]
        loader = get_dataloader(Xc, yc)
        print(f" Training client {client}...")
        train_local(local_model, loader)
        local_weights.append(local_model.state_dict())
        local_sizes.append(len(Xc))
    # FedAvg weights
    total_size = sum(local_sizes)
    avg_weights = OrderedDict()
    for k in local_weights[0].keys():
        avg_weights[k] = sum(local_weights[i][k].float() * local_sizes[i] / total_size for i in range(len(local_weights)))
    global_model.load_state_dict(avg_weights)
    global_weights = global_model.state_dict()

    acc = evaluate(global_model, test_loader)
    print(f"Test accuracy after round {r}: {acc*100:.2f}%")

print("Training finished.")

