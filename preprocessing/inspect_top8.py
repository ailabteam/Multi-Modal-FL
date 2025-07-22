import numpy as np

def inspect_top8_npz(npz_path):
    data = np.load(npz_path)
    print(f"File keys: {list(data.keys())}")
    
    X = data['X']
    y = data['y']
    client_ids = data['client_ids']
    
    print(f"X shape: {X.shape}")  # (samples, window_size, num_features)
    print(f"y shape: {y.shape}")
    print(f"client_ids shape: {client_ids.shape}")
    
    print("\nSample feature window (first sample):")
    print(X[0])
    
    print("\nSample label (first 10):")
    print(y[:10])
    
    print("\nSample client IDs (first 10):")
    print(client_ids[:10])

if __name__ == "__main__":
    npz_path = "data_pamap2_federated_top8_filtered.npz"  # Đường dẫn tới file top8 của bạn
    inspect_top8_npz(npz_path)

