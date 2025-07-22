import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.stats import mode

WINDOW_SIZE = 100
STEP_SIZE = 50
VALID_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
DATA_DIR = './PAMAP2_Dataset/Protocol/'

def sliding_windows(df):
    X = []
    y = []
    data = df.values
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i:i+WINDOW_SIZE, :]
        label_window = df['activityID'].values[i:i+WINDOW_SIZE]
        label_mode = mode(label_window, keepdims=True).mode[0]
        X.append(window[:, 3:])  # IMU data only (columns 4-54 zero-indexed as 3:)
        y.append(label_mode)
    return np.array(X), np.array(y)

def process_file(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.replace(to_replace='nan', value=np.nan, inplace=True)
    df.ffill(inplace=True)  # sửa deprecated warning
    df.bfill(inplace=True)  # sửa deprecated warning

    df.columns = ['timestamp', 'activityID', 'heart_rate'] + [f'col{i}' for i in range(4, 55)]
    df = df[df['activityID'].isin(VALID_ACTIVITIES)]
    df['subject'] = int(file_path.split('subject')[-1].split('.')[0])
    X, y = sliding_windows(df)
    clients = np.full((len(y),), df['subject'].iloc[0])
    return X, y, clients

def main():
    X_all = []
    y_all = []
    client_ids_all = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith('.dat'):
            print(f"Processing {os.path.join(DATA_DIR, fname)}...")
            X, y, clients = process_file(os.path.join(DATA_DIR, fname))
            X_all.append(X)
            y_all.append(y)
            client_ids_all.append(clients)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    client_ids_all = np.concatenate(client_ids_all)

    print("Raw labels sample:", y_all[:20])
    print("Unique raw labels count:", len(np.unique(y_all)))

    # --- Lọc top 8 label phổ biến nhất ---
    counter_labels = Counter(y_all)
    top8_labels = [label for label, _ in counter_labels.most_common(8)]
    print("Top 8 labels:", top8_labels)

    # Lọc dữ liệu chỉ giữ top8 label
    mask_labels = np.isin(y_all, top8_labels)
    X_all = X_all[mask_labels]
    y_all = y_all[mask_labels]
    client_ids_all = client_ids_all[mask_labels]

    # --- Lọc top 8 client phổ biến nhất (theo số mẫu) ---
    counter_clients = Counter(client_ids_all)
    top8_clients = [client for client, _ in counter_clients.most_common(8)]
    print("Top 8 clients:", top8_clients)

    mask_clients = np.isin(client_ids_all, top8_clients)
    X_all = X_all[mask_clients]
    y_all = y_all[mask_clients]
    client_ids_all = client_ids_all[mask_clients]

    print("After filtering: samples =", X_all.shape[0])
    print("Unique labels after filtering:", np.unique(y_all))
    print("Unique clients after filtering:", np.unique(client_ids_all))

    # Encode label thành 0..7 theo top8_labels
    label_encoder = LabelEncoder()
    label_encoder.fit(top8_labels)
    y_all_encoded = label_encoder.transform(y_all)
    print("Encoded labels unique:", np.unique(y_all_encoded))

    # In mapping
    print("== Mapping activityID -> encoded label ==")
    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        print(f"ActivityID {original} => Encoded label {encoded}")

    print("Final feature shape:", X_all.shape)
    print("Final label shape:", y_all_encoded.shape)
    print("Final client_ids shape:", client_ids_all.shape)
    print("NaNs in features:", np.isnan(X_all).sum(), ", NaNs in labels:", np.isnan(y_all_encoded).sum())
    print("Number of clients (subjects):", len(np.unique(client_ids_all)))
    print("Unique client IDs:", np.unique(client_ids_all))

    # Lưu dữ liệu
    np.savez_compressed("data_pamap2_federated_top8.npz",
                        X=X_all, y=y_all_encoded, client_ids=client_ids_all)
    print("Saved data_pamap2_federated_top8.npz successfully.")

if __name__ == '__main__':
    main()

