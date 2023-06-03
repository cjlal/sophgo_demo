import torch
from torch import nn
from model import Res2Net
from Res2Net import Bottle2neck

import numpy as np

from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r'./State_Farm/train_link/train'
ROOT_TEST = r'./State_Farm/train_link/val'

# 将图像RGB三个通道的像素值分别减去0.5,再除以0.5.从而将所有的像素值固定在[-1,1]范围内
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 裁剪为224*224
    transforms.RandomVerticalFlip(),  # 随机垂直旋转
    transforms.ToTensor(),  # 将0-255范围内的像素转为0-1范围内的tensor
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 调用net里面的定义的网络模型， 如果GPU可用则将模型转到GPU
model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4).to(device)

# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义优化器（SGD）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 学习率每隔10epoch变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    tran_acc = current / n
    print('train_loss:' + str(train_loss))
    print('train_acc:' + str(tran_acc))
    return train_loss, tran_acc


# 定义测试函数
def val(dataloader, model, loss_fn):
    # 将模型转为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

        val_loss = loss / n
        val_acc = current / n
        print('val_loss:' + str(val_loss))
        print('val_acc:' + str(val_acc))
        return val_loss, val_acc


# 画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title("训练集和验证集精确度值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 100
min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t + 1}\n--------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重文件
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        print(f'save best model,第{t + 1}轮')
        torch.save(model.state_dict(), 'save_model/best_model.pth')
    # 保存最后的权重模型文件
    if t == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')
print('Done！')

# matplot_loss(loss_train, loss_val)
# matplot_acc(acc_train, acc_val)





