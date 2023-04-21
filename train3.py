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
import model3
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

from functools import partial

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

    best_filename = 'models/model3_best.tar'
    filename = 'models/model3_checkpoint.tar'
    train_dataset = trainData(path="train.npz",leng=0)
    test_dataset = testData(path="test.npz",leng=0)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    cur_best = None

    net=model3.Net()
    net.to(device)

    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    # torch.save(net.state_dict(), "./testsize.pth")
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    earlystopping =EarlyStopping('max', patience=6)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    for epoch in range(1001):
        running_loss = 0.0
        correct = 0
        total = 0
        losssum = 0
        counter = 0
        net.train()
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()  # 优化器清零
            inputs = inputs.to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 优化
            running_loss += loss.item()
            losssum += loss.item()
            if i % 200 == 199:
                # print('[%d %5d] acc: %.3f' % (epoch + 1, i + 1, ))

                running_loss = 0.0

                correct = 0
                total = 0
            counter = counter + 1
        print('[training:%d ] acc: %.3f  loss: %.3f' % (epoch + 1,  correct / total, losssum / total))


        net.eval()
        test_loss = 0
        # target_num = torch.zeros((1, 2))  # n_classes为分类任务类别数量
        # predict_num = torch.zeros((1, 2))
        # acc_num = torch.zeros((1, 2))
        test_preds = []
        test_trues = []
        with torch.no_grad():

            for i, data in enumerate(test_loader, 0):
                total = total + len(data[0])
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                # optimizer.zero_grad()  # 优化器清零
                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # info2 = info2.to(torch.float32)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                test_preds.extend(predicted.detach().cpu().numpy())
                test_trues.extend(labels.detach().cpu().numpy())

                correct += (predicted == labels).sum().item()
                loss = cirterion(outputs, labels)
                losssum += loss.item()
            print('[test: %d ] acc: %.3f  loss: %.3f' % (epoch + 1,  correct / total, losssum / total))
        scheduler.step(correct / total)
        earlystopping.step(correct / total)
        testacc=correct / total
        is_best = not cur_best or testacc > cur_best
        if is_best:
            cur_best = testacc

        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'test_loss': test_loss,
            "test_acc": testacc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "lr": optimizer.state_dict()['param_groups'][0]['lr'],
        }, is_best, filename, best_filename)
        if earlystopping.stop:
            print("End of Training because of early stopping at epoch {}".format(epoch))
            break
    print('finished training!')
    print("end")
