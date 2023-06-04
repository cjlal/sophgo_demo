import numpy as np
import scipy.io as scio
import os
from sklearn.model_selection import KFold
def pre_process_data(data, norm_dim):
    """
    :param data: np.array, original traffic data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        norm_base: list, [max_data, min_data], data of normalization base.
        norm_data: np.array, normalized traffic data.
    """
    norm_base = normalize_base(data, norm_dim)  # find the normalize base
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  # normalize data

    return norm_base, norm_data

def normalize_base(data, norm_dim):
    """
    :param data: np.array, original traffic data without normalization.
    :param norm_dim: int, normalization dimension.
    :return:
        max_data: np.array
        min_data: np.array
    """
    max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
    min_data = np.min(data, norm_dim, keepdims=True)
    return max_data, min_data


def normalize_data(max_data, min_data, data):
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, original traffic data without normalization.
    :return:
        np.array, normalized traffic data.
    """
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / base

    return normalized_data


def dataset(k):
    eeg = scio.loadmat("../data/" + "sameEEG_de_LDS1.mat")
    label = scio.loadmat("../data/" + "sameLabel_de_LDS1.mat")
    data = eeg["sameEEG_de_LDS1"]
    label = label["sameLabel_de_LDS1"]

    count0 = 0
    count1 = 0
    count2 = 0
    label = label.reshape(-1, 1)
    label_A = [int(i) for i in label]  # 将标签数据转换为整型
    label = np.array(label_A).reshape(-1, 1)
    for i in range(len(label)):
        if label[i] == 0:
            count0 += 1
        elif label[i] == 1:
            count1 += 1
        elif label[i] == 2:
            count2 += 1
    print("Sum 0:", count0, "1:", count1, "2:", count2)  # ,"4:",count4

    norm_data = data[:]
    # norm_data = np.nan_to_num(norm_data,0)
    label = np.reshape(label, (-1))
    # subject-dependent
    x_train = norm_data[k * 3394:k * 3394+2010]  # 146
    y_train = label[k * 3394:k * 3394+2010]

    # subject-independent
    # if k == 0:
    #     x_train = norm_data[(k + 1) * 3394: (k + 15) * 3394]  # 146
    #     y_train = label[(k + 1) * 3394: (k + 15) * 3394]
    # elif k == 14:
    #     x_train = norm_data[0 * 3394: k * 3394]  # 146
    #     y_train = label[0 * 3394: k * 3394]
    # else:
    #     x_train1 = norm_data[0 * 3394: k * 3394]  # 146
    #     y_train1 = label[0 * 3394: k * 3394]
    #     x_train2 = norm_data[(k + 1) * 3394:15 * 3394]  # 146
    #     y_train2 = label[(k + 1) * 3394: 15 * 3394]
    #     x_train = np.append(x_train1, x_train2, axis=0)
    #     y_train = np.append(y_train1, y_train2, axis=0)

    # subject-dependent
    x_test = norm_data[k*3394+2010:(k+1)*3394]
    y_test = label[k*3394+2010:(k+1)*3394]

    # subject-independent
    # x_test = norm_data[k * 3394:(k + 1) * 3394]
    # y_test = label[k * 3394:(k + 1) * 3394]

    # 训练集归一化
    for channel in range(0, x_train.shape[1]):
        train_channel_data = x_train[:, channel, :]
        test_channel_data = x_test[:, channel, :]
        eeg_mean = np.mean(train_channel_data, axis=0)
        eeg_std = np.std(train_channel_data, axis=0)
        train_eeg_norm = (train_channel_data - eeg_mean) / eeg_std
        test_eeg_norm = (test_channel_data - eeg_mean) / eeg_std
        x_train[:, channel, :] = train_eeg_norm
        x_test[:, channel, :] = test_eeg_norm
        # eeg_max = np.max(x_train, axis=0)
        # eeg_min = np.min(x_train,axis=0)
        # eeg_norm = (x_train-eeg_min)/(eeg_max-eeg_min)
    Train_count0 = 0
    Train_count1 = 0
    Train_count2 = 0
    for i in range(len(y_train)):
        if y_train[i] == 0:
            Train_count0 += 1
        elif y_train[i] == 1:
            Train_count1 += 1
        elif y_train[i] == 2:
            Train_count2 += 1
    print("Train: 0:", Train_count0, "1:", Train_count1, "2:", Train_count2)

    Test_count0 = 0
    Test_count1 = 0
    Test_count2 = 0
    for i in range(len(y_test)):
        if y_test[i] == 0:
            Test_count0 += 1
        elif y_test[i] == 1:
            Test_count1 += 1
        elif y_test[i] == 2:
            Test_count2 += 1
    print("Test: 0:", Test_count0, "1:", Test_count1, "2:", Test_count2)
    # scio.savemat("../data/example_train_data.mat", {"train_data": [x_train,Train_count0, Train_count1, Train_count2]})
    # scio.savemat("../data/example_train_label.mat", {"train_label": y_train})
    # scio.savemat("../data/example_test_data.mat", {"test_data": x_test})
    # scio.savemat("../data/example_test_label.mat", {"test_label": y_test})
    return x_train, x_test, y_train, y_test, Train_count0, Train_count1, Train_count2

if __name__ == "__main__":
    dataset(0)
