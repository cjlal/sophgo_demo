import time
import shutil
from torch.utils import data
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
from model.GlobalTransformer import GlobalTransformer
from example_read_data import dataset
import numpy as np
import torch.nn.functional as F
from model.pytorchtools import EarlyStopping
import random
seed = 1 #1、50、100、500、1000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子`

acclist = []

class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:logists
        labels:
        """
        preds = F.softmax(preds,dim=1)
        eps = 1e-7
        target = self.one_hot(preds.size(1), labels)
        ce = -1 * torch.log(preds+eps) * target
        floss = torch.pow((1-preds), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0),num)).to(labels.device)
        one[range(labels.size(0)),labels] = 1
        return one

class RandomDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.data = x
        self.label = y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len

best_acc1 = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

model_save_path = "./results/GlobalTransformer/model/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

f_name = "de"
# print(device)

def main(k):
    global best_acc1
    x_train, x_test, y_train, y_test, trainNum0, trainNum1, trainNum2= dataset(k)

    xdim = x_train.shape

    model = GlobalTransformer(in_c=5, num_T_head=4, att_dim=64, hidden=1024, num_encoder=6, graph_size=xdim[1])

    model = model.to(device)

    w1 = (trainNum1+trainNum2) / (trainNum2+trainNum1 + trainNum0)
    w2 =  (trainNum2+trainNum0) / (trainNum2+trainNum1 + trainNum0)
    w3 =  (trainNum1+trainNum0)  / (trainNum2 + trainNum1 + trainNum0)

    # criterion = nn.CrossEntropyLoss()
    criterion = Focal_Loss(torch.tensor([w1, w2,w3]).to(device))

    # optimize
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, 0.0001, weight_decay=0.1)#0.05,0.001,0.002

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(dataset=RandomDataset(x_train, y_train),
                                               batch_size=32, shuffle=True, drop_last=False,num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=RandomDataset(x_test, y_test),
                                             batch_size=32, shuffle=False, drop_last=False)

    Accuracy_list_train = []
    Accuracy_list_val = []
    Loss_list_train = []
    Loss_list_val = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(500):
        # break
        adjust_learning_rate(optimizer, epoch, 0.0001, [20,50,80,100,120,150,180,200,250,280])#[50, 80]
        # train for one epoch
        acc1, loss = train(train_loader, model, criterion, optimizer, epoch)
        Loss_list_train.append(loss)
        Accuracy_list_train.append(acc1.item())

        # evaluate on validation set
        acc1_val, loss_val = validate(val_loader, model, criterion)
        Loss_list_val.append(loss_val)
        Accuracy_list_val.append(acc1_val.item())

        is_best = acc1_val > best_acc1
        best_acc1 = max(acc1_val, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, "_{}_".format(f_name)+str(k))
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(best_acc1)
    print(Accuracy_list_train)
    print(Accuracy_list_val)
    best_acc1 = 0

def save_checkpoint(state, is_best, i):
    filename = 'checkpoint' + i + '.pth.tar'
    path = "{}/".format(model_save_path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path + filename, path + 'model_best' + i + '.pth.tar')

def train(train_loader,model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    # 临时添加
    with torch.autograd.set_detect_anomaly(True):
        for i, (data, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            target = target.squeeze()
            data = data.to(device)
            target = target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target.long())
            # print("loss:",loss)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target.long(), topk=(1, 2))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:  # 10
                progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (data, target) in enumerate(val_loader):
            target = target.squeeze()
            data = data.to(device)
            target = target.to(device)
            # compute output
            output = model(data)
            loss = criterion(output, target.long())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target.long(), topk=(1, 2))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Loss {losses.avg:.3f}'
              .format(top1=top1, losses=losses))

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr, schedule):
    """Decay the learning rate based on schedule"""
    lr = lr
    for milestone in schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main(0)
