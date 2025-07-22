import os
import numpy as np
import pandas as pd
from scipy.stats import mode

# Đường dẫn thư mục chứa file .dat
DATA_DIR = './PAMAP2_Dataset/Protocol/'

# Các cột bạn định nghĩa (có thể mở rộng, hiện tại 52 cột cảm biến + 2 cột khác)
columns = [
    'timestamp', 'activity_id', 'heart_rate',
    # Sensor IMU on hand (3x accel, 3x gyro, 3x mag, 4x quat, temp)
    'hand_acc_x', 'hand_acc_y', 'hand_acc_z',
    'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'hand_mag_x', 'hand_mag_y', 'hand_mag_z',
    'hand_quat_w', 'hand_quat_x', 'hand_quat_y', 'hand_quat_z',
    'hand_temp',
    # Sensor IMU on chest
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
    'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
    'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
    'chest_quat_w', 'chest_quat_x', 'chest_quat_y', 'chest_quat_z',
    'chest_temp',
    # Sensor IMU on ankle
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z',
    'ankle_mag_x', 'ankle_mag_y', 'ankle_mag_z',
    'ankle_quat_w', 'ankle_quat_x', 'ankle_quat_y', 'ankle_quat_z',
    'ankle_temp'
]

LABEL_COL = 'activity_id'

WINDOW_SIZE = 128
STEP_SIZE = 64

def get_columns(total_cols=114):
    base_cols = [
        'timestamp', 'activity_id', 'heart_rate'
    ]
    sensor_cols = [f'sensor{i}' for i in range(total_cols - len(base_cols))]
    return base_cols + sensor_cols

def sliding_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    X = []
    y = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start:start + window_size]
        labels = window[LABEL_COL]

        if len(labels) == 0:
            continue

        mode_result = mode(labels)
        # Kiểm tra mode_result.mode có phải mảng không
        if hasattr(mode_result.mode, '__len__'):
            label_mode = mode_result.mode[0]
        else:
            label_mode = mode_result.mode  # scalar

        X.append(window.drop(columns=[LABEL_COL]).values.flatten())
        y.append(label_mode)

    return np.array(X), np.array(y)


def process_file(filepath):
    total_cols = 114
    cols = get_columns(total_cols)
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols, low_memory=False)

    # Xử lý NaN bằng nội suy, ffill, bfill
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Lọc nhãn không hợp lệ
    df = df[df[LABEL_COL].notna()]
    df = df[df[LABEL_COL] != -1]

    X, y = sliding_windows(df)
    return X, y

def process_all(data_dir=DATA_DIR):
    all_X = []
    all_y = []

    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat')])
    for f in files:
        print(f'Processing {f}')
        X, y = process_file(f)
        if len(X) == 0:
            print(f'Warning: no data extracted from {f}, skipping.')
            continue
        all_X.append(X)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.hstack(all_y)

    print(f'Final feature shape: {X_all.shape}, label shape: {y_all.shape}')
    print(f'NaNs in labels: {np.sum(np.isnan(y_all))}')

    np.savez_compressed('data_pamap2.npz', X=X_all, y=y_all)
    print('Saved processed data to data_pamap2.npz')

if __name__ == '__main__':
    process_all()

