

import pickle
import numpy as np
from parameters import Hyperparams as hp
import matplotlib.pyplot as plt

def read_data(data_path):
    try:
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"错误：文件 {data_path} 未找到。")
    except Exception as e:
        print(f"读取数据时出现错误：{e}")
    return None

def inverse_data(datas,scarla_):
    return scarla_.inverse_transform(np.squeeze(datas))

def draw_gt_pred(gt,pred):
    """

    Args:
        gt: ground truth
        pred: predictions

    Returns:a pc between gt and pred

    """
    plt.figure(figsize=(20,8),dpi=120)
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    # plt.xticks(np.arange(0,len(gt),288))#per day to visualization
    plt.tick_params(labelsize=18)
    plt.xlim(0,len(gt)+4)
    plt.ylim(0,max(np.max(gt),np.max(pred))+50)
    plt.plot(gt,'-',color='deepskyblue',linewidth=1)
    plt.plot(pred, '-.',color='orangered',linewidth=1.5)
    plt.show()


def print_muti_error(pred, test_label):
    all = []
    # 计算 MAE
    muti_er = np.mean(np.abs(pred - test_label), axis=0)
    all.append(muti_er)

    # 计算 MSE
    muti_er_mse = np.mean(np.square(pred - test_label), axis=0)
    all.append(muti_er_mse)

    # 计算 MAPE
    muti_er_mape = np.mean(np.abs(pred - test_label) / test_label, axis=0)
    all.append(muti_er_mape)

    # 计算 RMSE
    muti_er_rmse = np.sqrt(muti_er_mse)
    all.append(muti_er_rmse)

    # 选择指定的索引
    selected_indices = [2, 5, 8, 11]
    indices = [index for index in selected_indices if index < len(muti_er)]

    # 输出结果
    print('MAE is:', ', '.join(f'{muti_er[idx]:.2f}' for idx in indices))
    print('MSE is:', ', '.join(f'{muti_er_mse[idx]:.2f}' for idx in indices))
    print('RMSE is:', ', '.join(f'{muti_er_rmse[idx]:.2f}' for idx in indices))
    print('MAPE is:', ', '.join(f'{muti_er_mape[idx]:.4f}' for idx in indices))


    return all



