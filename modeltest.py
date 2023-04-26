import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from functools import partial
import cv2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=15, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=15, out_channels=9, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(26244 , 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(44944, 4096)
        # self.fc2 = nn.Linear(4096, 512)
        # self.fc3 = nn.Linear(512,30)
    def forward(self, x):
        x = self.conv1(x)  # input(3, 32, 32)  output(16, 28, 28)
        x = self.relu(x)  # 激活函数
        x = self.maxpool1(x)  # output(16, 14, 14)
        x = self.conv2(x)  # output(32, 10, 10)
        x = self.relu(x)  # 激活函数
        x = self.maxpool2(x)  # output(32, 5, 5)
        x = torch.flatten(x, start_dim=1)  # output(32*5*5) N代表batch_size
        x = self.fc1(x)  # output(120)
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # output(84)
        x = self.relu(x)  # 激活函数
        x = self.fc3(x)  # output(num_classes)

        return x
class trainData(Dataset):
    def __init__(self, path="/home/mao/23Spring/cars/racing_car_data/record/train/",leng=0):
        data=np.load(path)
        self.obs=np.load('train.npz')["obs"]
        self.lbl=np.load('train.npz')["lbl"]

    # train_data =
    # x_train = train_data
    # y_train = train_data['lbl']
    # test_data = np.load("test.npz")
    # x_test = test_data["obs"]
    # y_test = test_data['lbl']
    def __getitem__(self, index):

        return self.obs[index].reshape(1,224,224),self.lbl[index]
    def __len__(self):
        return len(self.lbl)

class EarlyStopping(object): # pylint: disable=R0902
    """
    Gives a criterion to stop training when a given metric is not
    improving anymore
    Args:
        mode (str): One of `min`, `max`. In `min` mode, training will
            be stopped when the quantity monitored has stopped
            decreasing; in `max` mode it will be stopped when the
            quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which training is stopped. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only stop learning after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.

    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """ Updates early stopping state """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        """ Should we stop learning? """
        return self.num_bad_epochs > self.patience


    def _cmp(self, mode, threshold_mode, threshold, a, best): # pylint: disable=R0913, R0201
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ Returns early stopping state """
        return {key: value for key, value in self.__dict__.items() if key != 'is_better'}

    def load_state_dict(self, state_dict):
        """ Loads early stopping state """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)




class testData(Dataset):
    def __init__(self, path="/home/mao/23Spring/cars/racing_car_data/record/train/", leng=0):
        data = np.load(path)
        self.obs = np.load('test.npz')["obs"]
        self.lbl = np.load('test.npz')["lbl"]

    # train_data =
    # x_train = train_data
    # y_train = train_data['lbl']
    # test_data = np.load("test.npz")
    # x_test = test_data["obs"]
    # y_test = test_data['lbl']
    def __getitem__(self, index):
        return self.obs[index].reshape(1,224,224), self.lbl[index]

    def __len__(self):
        return len(self.lbl)

def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
if __name__ == '__main__':
    device='cuda'
    acts_space=[(-1, 0.05), (-1, 0.15000000000000002), (-1, 0.25), (-1, 0.35000000000000003), (-1, 0.45), (-1, 0.55),
     (-1, 0.6500000000000001), (-1, 0.7500000000000001), (-1, 0.8500000000000001), (-1, 0.9500000000000001), (0, 0.05),
     (0, 0.15000000000000002), (0, 0.25), (0, 0.35000000000000003), (0, 0.45), (0, 0.55), (0, 0.6500000000000001),
     (0, 0.7500000000000001), (0, 0.8500000000000001), (0, 0.9500000000000001), (1, 0.05), (1, 0.15000000000000002),
     (1, 0.25), (1, 0.35000000000000003), (1, 0.45), (1, 0.55), (1, 0.6500000000000001), (1, 0.7500000000000001),
     (1, 0.8500000000000001), (1, 0.9500000000000001)]
    best_filename = 'models/model2_simu.tar'
    filename = 'models/model2_checkpoint.tar'
    # train_dataset = trainData(path="train.npz",leng=0)
    # test_dataset = testData(path="test.npz",leng=0)
    #
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    cur_best = None
    state=torch.load(best_filename)
    net=Net()
    net.load_state_dict(state['state_dict'])
    net.to(device)

    #gpu speed:0.00039920692443847655
    #cpu speed:0.0019628223896026612

    # torch.save(net.state_dict(), "./testsize.pth")

    time0=time.time()
    for epoch in range(5):
        time0 = time.time()

        running_loss = 0.0
        correct = 0
        total = 0
        losssum = 0
        counter = 0
        net.eval()

        train_data=[]
        datanames = os.listdir('/home/mao/23Spring/cars/donkey_train/data/images')
        for dataname in datanames:
            # if os.path.splitext(dataname)[1] == '.json':
            #     # 目录下包含.json的文件
            #     train_lbl.append(logpath + dataname)
            if os.path.splitext(dataname)[1] == '.jpg':
                train_data.append('/home/mao/23Spring/cars/donkey_train/data/images/' + dataname)
        train_data = sorted(train_data, key=lambda file: os.path.getctime(file))

        for i in range(0, 10000):
            inputs=torch.tensor(cv2.imread(train_data[i]))
            inputs=inputs.to(device)
            outputs = net(inputs.float().view(1,3,224,224))
            print(outputs)


        print(time.time()-time0)
        print((time.time()-time0)/total)

