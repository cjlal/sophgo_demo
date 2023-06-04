from torch.utils import data
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset
from model.DGCNN import DGCNN
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import scipy.io as scio

acclist = []

class RandomDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        self.data = x
        self.label = y

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
save_path = "./results/DGCNN/SEED/DE"
model_save_path = "./results/DGCNN/model/model_best_de_0.pth.tar"

if not os.path.exists(save_path):
    os.makedirs(save_path)

f_name = "de"

def main():
    x_test = scio.loadmat("./data/example_test_data.mat")
    x_test = x_test["test_data"]
    y_test = scio.loadmat("./data/example_test_label.mat")
    y_test = y_test["test_label"]
    xdim = x_test.shape
    model = DGCNN(xdim=xdim, k_adj=5, num_out=64, nclass=3, dropout=0.5, device=device)
    model = model.to(device)

    x_test = torch.from_numpy(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.from_numpy(y_test)
    y_test = y_test.reshape(y_test.shape[1], 1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    val_loader = torch.utils.data.DataLoader(dataset=RandomDataset(x_test, y_test),
                                             batch_size=32, shuffle=False, drop_last=False)

    print_test(val_loader, model, model_save_path)

def print_test(test_loader, model,model_save_path):
    if os.path.isfile(model_save_path):
        print("=> loading checkpoint '{}'".format(model_save_path))
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(model_save_path))

    targetlist, scorelist, predlist = Test(test_loader, model)

    # print confusion matrix
    true_labels = torch.Tensor(targetlist)
    pre_labels = torch.Tensor(predlist)
    acc = accuracy_score(true_labels, pre_labels)
    print('Accuracy =', acc * 100.0, "%")
    matrix = confusion_matrix(true_labels, pre_labels)
    print('Confusion Matrix:')
    print(np.matrix(matrix))

    # print classification report
    target_names = ['1', '2','3']
    print('Classification Report:')
    print(classification_report(true_labels, pre_labels,
                                target_names=target_names, digits=5))
    report = classification_report(true_labels, pre_labels,
                                   target_names=target_names, digits=5, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df2 = pd.DataFrame([matrix[0], matrix[1],matrix[2]], index=["matrix0:", "matrix1:","matrix2:"])
    df.to_csv("{}/{}_result.csv".format(save_path,f_name), index=True)
    df2.to_csv("{}/{}_matrix.csv".format(save_path,f_name), index=True)

def Test(test_loader, model):
    model.eval()
    with torch.no_grad():
        predlist = []
        scorelist = []
        targetlist = []
        # Predict
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            # compute output
            output = model(data)

            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

    return targetlist, scorelist, predlist

if __name__ == '__main__':
    main()
