"""
   @Author: Weiyang Jiang
   @Date: 2022-01-05 16:22:28
"""
import math
import os
import time

import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"
data = pd.read_csv("test_1.csv")

# drop line with 0 value
data[data == 0] = np.nan
data = data.dropna()


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X = X.to(device)
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    x = torch.mm(X.double(), components)
    return x


data = np.array(data)
feature_data = torch.from_numpy(data[:, 1:]).float()
feature_data = PCA_svd(feature_data, 4)

split_point = round(data.shape[0] * 0.8)
train_data_feature = feature_data[:split_point, :]
test_data_feature = feature_data[split_point:, :]
train_data_label = data[:split_point, :1]
test_data_label = data[split_point:, :1]


class DietengDataset(Dataset):
    def __init__(self, feature, label):
        self.data = feature.float()
        # Normalize features
        self.data = \
            (self.data - self.data.mean(dim=0, keepdim=True)) \
            / self.data.std(dim=0, keepdim=True)

        self.label = torch.from_numpy(label).float()

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class DietengNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DietengNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        x = self.feature(x)
        return x

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def cal_R2(pred_list, target_list):
    xBar = np.mean(pred_list)
    yBar = np.mean(target_list)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(pred_list)):
        diffXXBar = pred_list[i] - xBar
        diffYYBar = target_list[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r_squared = (SSR / SST) ** 2
    return r_squared[0]

    # fix random seed


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)


def save_csv(loss, R2):
    str_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    loss = math.sqrt(loss)
    with open("models/loss.csv", "a+") as file:
        file.write("\n{}, {}, {}".format(str_time, loss, R2))


# the path where checkpoint saved
def model_path(acc, lr):
    return f"model_{acc}_{lr}.ckpt"


def train(train_set, test_set, model, config, device):
    n_epochs = config['n_epochs']

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optimizer_hparas'])

    min_mse = 99999.
    loss_record = {'train': [], 'test': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # set model to training mode
        for x, y in train_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device
            pred = model(x)  # forward pass
            mse_loss = model.cal_loss(pred, y)  # compute loss

            """ 
            # adding L1 Norm
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            mse_loss = mse_loss + 0.01 * regularization_loss
            # adding L1 Norm  
            """

            mse_loss.backward()  # compute gradient
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test model
        test_mse = test(test_set, model, device)

        if test_mse < min_mse:
            # Save model if model improved
            min_mse = test_mse
            print('Saving model (epoch = {:4d}, loss = {})'
                  .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            # if the valid loss of model continues to decrease in the process of 200 epochs of training,
            # stop the training program.
            early_stop_cnt += 1

        epoch += 1
        loss_record['test'].append(test_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def test(test_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in test_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass
            mse_loss = model.cal_loss(pred, y)  # compute loss
            """
            # adding L1 Norm
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
            mse_loss = mse_loss + 0.01 * regularization_loss
            # adding L1 Norm
            """
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(test_set.dataset)  # compute averaged loss
    return total_loss


def plot_learning_curve(loss_record, title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['test'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], label='train')
    plt.plot(x_2, loss_record['test'], label='test')
    plt.ylim(0.0, 0.012)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=1.2, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
    R2 = cal_R2(preds, targets)
    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.01, lim], [-0.01, lim], c='b')
    plt.xlim(-0.01, lim)
    plt.ylim(-0.01, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
    return R2


def main():
    config = {
        'n_epochs': 3000,  # maximum number of epochs
        'batch_size': 64,  # mini-batch size for dataloader
        'optimizer': 'SGD',  # optimization algorithm
        'optimizer_hparas': {  # hyper-parameters for the optimizer
            'lr': 0.00065,
            "weight_decay": 1e-4,  # Using for L2 Norm
            'momentum': 0.9
        },
        'early_stop': 200,  # early stopping epochs
        'save_path': 'models/model.pth'
    }
    train_dataset = DietengDataset(train_data_feature, train_data_label)
    test_dataset = DietengDataset(test_data_feature, test_data_label)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    model = DietengNetwork(4).to(device)
    # summary(model.cuda(), (1, 4))
    # start training

    model_loss, loss_record = train(train_loader, test_loader, model, config, device)

    plot_learning_curve(loss_record, "Network")
    R2 = plot_pred(test_loader, model, device)
    save_csv(model_loss, R2)


if __name__ == '__main__':
    main()

