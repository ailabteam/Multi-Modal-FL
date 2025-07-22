import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder

WINDOW_SIZE = 100
STEP_SIZE = 50
VALID_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24]
DATA_DIR = './PAMAP2_Dataset/Protocol/'

TOP8_LABELS = [17, 4, 1, 3, 7, 2, 16, 6]
TOP8_CLIENTS = [108, 105, 102, 106, 104, 107, 101, 103]  # ban đầu có client 103 nhưng sẽ loại bỏ sau

def sliding_windows(df):
    X = []
    y = []
    data = df.values
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i:i+WINDOW_SIZE, :]
        label_window = df['activityID'].values[i:i+WINDOW_SIZE]
        label_mode = Counter(label_window).most_common(1)[0][0]
        X.append(window[:, 3:])  # IMU data only
        y.append(label_mode)
    return np.array(X), np.array(y)

def process_file(file_path):
    import pandas as pd
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.replace(to_replace='nan', value=np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df.columns = ['timestamp', 'activityID', 'heart_rate'] + [f'col{i}' for i in range(4, 55)]
    df = df[df['activityID'].isin(VALID_ACTIVITIES)]
    df['subject'] = int(file_path.split('subject')[-1].split('.')[0])
    X, y = sliding_windows(df)
    clients = np.full((len(y),), df['subject'].iloc[0])
    return X, y, clients

def main():
    X_all = []
    y_all = []
    client_ids = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith('.dat'):
            X, y, clients = process_file(os.path.join(DATA_DIR, fname))
            X_all.append(X)
            y_all.append(y)
            client_ids.append(clients)

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    client_ids = np.concatenate(client_ids)

    # Filter by top 8 labels
    mask_label = np.isin(y_all, TOP8_LABELS)
    X_all = X_all[mask_label]
    y_all = y_all[mask_label]
    client_ids = client_ids[mask_label]

    # Filter to top 8 clients first (optional)
    mask_client = np.isin(client_ids, TOP8_CLIENTS)
    X_all = X_all[mask_client]
    y_all = y_all[mask_client]
    client_ids = client_ids[mask_client]

    # Kiểm tra client có đủ 8 nhãn hay không
    valid_clients = []
    for c in np.unique(client_ids):
        labels_c = np.unique(y_all[client_ids == c])
        if set(labels_c) == set(TOP8_LABELS):
            valid_clients.append(c)
        else:
            print(f"Client {c} thiếu nhãn, loại bỏ: có nhãn {labels_c}")

    # Lọc lại dữ liệu chỉ giữ client có đủ 8 nhãn
    mask_valid_client = np.isin(client_ids, valid_clients)
    X_all = X_all[mask_valid_client]
    y_all = y_all[mask_valid_client]
    client_ids = client_ids[mask_valid_client]

    print(f"Sau lọc, số samples: {len(y_all)}")
    print(f"Số client còn lại: {len(valid_clients)} - {valid_clients}")
    print(f"Nhãn còn lại: {np.unique(y_all)}")

    # Encode lại nhãn (LabelEncoder)
    le = LabelEncoder()
    y_all_encoded = le.fit_transform(y_all)
    print("== Mapping activityID -> encoded label ==")
    for original, encoded in zip(le.classes_, le.transform(le.classes_)):
        print(f"ActivityID {original} => Encoded label {encoded}")

    # Save file .npz
    np.savez("data_pamap2_federated_top8_filtered.npz", X=X_all, y=y_all_encoded, client_ids=client_ids)
    print("Saved data_pamap2_federated_top8_filtered.npz successfully.")

if __name__ == '__main__':
    main()

