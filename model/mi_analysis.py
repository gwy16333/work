# evaluate/feature_decoupling/mi_analysis.py
from sklearn.feature_selection import mutual_info_regression
import numpy as np


def calculate_mi(F_s, F_t, n_samples=10000):
    # 展平特征
    F_s_flat = F_s.reshape(-1, F_s.shape[-1])  # (N_samples*N_nodes, C_s)
    F_t_flat = F_t.reshape(-1, F_t.shape[-1])  # (N_samples*T_seq, C_t)

    # 随机采样
    idx = np.random.choice(F_s_flat.shape[0], n_samples, replace=False)
    F_s_sampled = F_s_flat[idx]
    F_t_sampled = F_t_flat[idx]

    # 计算MI
    mi_values = []
    for i in range(F_s_sampled.shape[1]):
        mi = mutual_info_regression(F_t_sampled, F_s_sampled[:, i])
        mi_values.append(mi.mean())

    return np.mean(mi_values)


if __name__ == "__main__":
    # 加载特征
    data = np.load("features.npz")
    F_s, F_t = data["F_s"], data["F_t"]

    # 计算并打印MI
    mi = calculate_mi(F_s, F_t)
    print(f"Mutual Information: {mi:.4f}")