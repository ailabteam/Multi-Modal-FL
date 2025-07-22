import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

# --- Config ---
DATA_PATH = 'data_pamap2_federated_top8_filtered.npz'
BATCH_SIZE = 64
LOCAL_EPOCHS = 3  # tăng local epochs lên 3
ROUNDS = 100
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load dữ liệu ---
data = np.load(DATA_PATH)
X = data['X']  # (N, 100, 52)
y = data['y']  # (N,)
client_ids = data['client_ids']  # (N,)

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}, clients shape: {client_ids.shape}")

# --- MLP model ---
class MLP(nn.Module):
    def __init__(self, input_dim=5200, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --- Chuẩn bị dữ liệu từng client ---
# Flatten X: (N, 100, 52) -> (N, 5200)
X_flat = X.reshape(X.shape[0], -1)
clients_data = {}
for client in np.unique(client_ids):
    idxs = np.where(client_ids == client)[0]
    clients_data[client] = (X_flat[idxs], y[idxs])

# --- Helper functions ---
def get_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_local(model, dataloader, epochs=LOCAL_EPOCHS):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

def average_weights(w_list, weights):
    avg_w = OrderedDict()
    for k in w_list[0].keys():
        avg_w[k] = torch.stack([w[k].float() * weights[i] for i, w in enumerate(w_list)], 0).sum(0)
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

# --- Tạo dataloader test (lấy 10% dữ liệu random) ---
np.random.seed(42)
all_indices = np.arange(len(X_flat))
np.random.shuffle(all_indices)
split_idx = int(0.9 * len(all_indices))
train_idx = all_indices[:split_idx]
test_idx = all_indices[split_idx:]

X_train, y_train = X_flat[train_idx], y[train_idx]
X_test, y_test = X_flat[test_idx], y[test_idx]
test_loader = get_dataloader(X_test, y_test, shuffle=False)

# --- Khởi tạo global model ---
global_model = MLP().to(DEVICE)
global_weights = global_model.state_dict()

# --- Bắt đầu vòng FL ---
for r in range(1, ROUNDS+1):
    print(f"\n--- Round {r} ---")
    local_weights = []
    local_sizes = []
    for client in clients_data.keys():
        local_model = MLP().to(DEVICE)
        local_model.load_state_dict(global_weights)
        Xc, yc = clients_data[client]
        loader = get_dataloader(Xc, yc)
        train_local(local_model, loader, epochs=LOCAL_EPOCHS)
        local_weights.append(local_model.state_dict())
        local_sizes.append(len(Xc))

    total_size = sum(local_sizes)
    weights = [size / total_size for size in local_sizes]
    avg_weights = average_weights(local_weights, weights)

    global_model.load_state_dict(avg_weights)
    global_weights = global_model.state_dict()

    acc = evaluate(global_model, test_loader)
    print(f"Test accuracy after round {r}: {acc*100:.2f}%")

print("Training finished.")

