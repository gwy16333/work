# evaluate/feature_decoupling/utils.py
import numpy as np


def extract_features(model, dataloader, save_path):
    model.eval()
    all_F_s, all_F_t = [], []

    with torch.no_grad():
        for x, _ in dataloader:  # 假设数据加载器返回 (input, target)
            _, F_s, F_t = model(x)
            all_F_s.append(F_s.cpu().numpy())
            all_F_t.append(F_t.cpu().numpy())

    # 合并并保存
    F_s = np.concatenate(all_F_s, axis=0)  # shape=(N_samples, N_nodes, C_s)
    F_t = np.concatenate(all_F_t, axis=0)  # shape=(N_samples, T_seq, C_t)
    np.savez(save_path, F_s=F_s, F_t=F_t)