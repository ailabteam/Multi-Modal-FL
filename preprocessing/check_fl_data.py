import numpy as np
from collections import Counter

def check_data(path):
    data = np.load(path)
    X = data['X']
    y = data['y']

    labels = y[:, 0].astype(int)
    clients = y[:, 1].astype(int)

    print(f"Total samples: {len(labels)}")
    print(f"Unique encoded classes: {np.unique(labels)}")
    print(f"Unique clients: {np.unique(clients)}")

    global_label_counter = Counter(labels)
    print("\n== Global label distribution ==")
    for lbl, cnt in sorted(global_label_counter.items()):
        print(f"Label {lbl}: {cnt} samples")

    for cid in np.unique(clients):
        mask = clients == cid
        client_labels = labels[mask]
        label_dist = Counter(client_labels)
        print(f"\nClient {cid} - {mask.sum()} samples - {len(label_dist)} labels")
        for lbl, cnt in sorted(label_dist.items()):
            print(f"  Label {lbl}: {cnt} samples")

check_data("data_pamap2_federated.npz")

