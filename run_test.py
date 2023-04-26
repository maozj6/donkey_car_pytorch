

from torch.utils.data import Dataset
import numpy as np

from functools import partial


from Adafruit_GPIO import I2C

import torch.nn as nn

import time
import cv2
import torch

import Adafruit_PCA9685
import donkeycar as dk


PCA9685_I2C_BUSNUM = 1
PCA9685_I2C_ADDR = 0x40
def get_bus():
    return PCA9685_I2C_BUSNUM


I2C.get_default_bus = get_bus

pwm = Adafruit_PCA9685.PCA9685(address=PCA9685_I2C_ADDR)
pwm.set_pwm_freq(60)  # frequence of PWM

STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 275  # pwm value for full left steering
STEERING_RIGHT_PWM = 105  # pwm value for full right steering

THROTTLE_CHANNEL = 0

THROTTLE_FORWARD_PWM = 430  # 430 before      #pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370  # pwm value for no movement
THROTTLE_REVERSE_PWM = 300  # pwm value for max reverse throttle

"""init steering and ESC"""
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_FORWARD_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_REVERSE_PWM))
time.sleep(0.1)
pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
time.sleep(0.1)

def gstreamer_pipeline(capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21,
                       flip_method=0):
    return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
        capture_width, capture_height, framerate, flip_method, output_width, output_height)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(26244, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, 1)

        self.sgm=nn.Sigmoid()

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

        x=self.sgm(x)
        return x

class trainData(Dataset):
    def __init__(self, path="/home/mao/23Spring/cars/racing_car_data/record/train/",leng=0):
        data=np.load(path)
        self.obs=data["obs"]
        self.lbl=data["lbl"]

    # train_data =
    # x_train = train_data
    # y_train = train_data['lbl']
    # test_data = np.load("test.npz")
    # x_test = test_data["obs"]
    # y_test = test_data['lbl']
    def __getitem__(self, index):

        return self.obs[index].reshape(3,224,224),self.lbl[index]
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
        self.obs = data["obs"]
        self.lbl = data["lbl"]

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
    best_filename = 'model2_newbest.tar'

    state = torch.load(best_filename)

    net = Net()
    net.load_state_dict(state['state_dict'])
    net.to(device)


    net=Net()
    net.to(device)

    w = 224
    h = 224
    running = True
    frame = None
    flip_method = 6
    capture_width = 224
    capture_height = 224
    framerate = 15

    camera = cv2.VideoCapture(
        gstreamer_pipeline(
            capture_width=capture_width,
            capture_height=capture_height,
            output_width=w,
            output_height=h,
            framerate=framerate,
            flip_method=flip_method),
        cv2.CAP_GSTREAMER)

    ret = True
    count = 0

    while count<800:
        count+=1
        running_loss = 0.0
        correct = 0
        total = 0
        losssum = 0
        counter = 0
        net.eval()
        ret, img = camera.read()
        print(ret)

        # img = cv2.imread("test_img/" + str(i)+".jpg")
        img=img.reshape(3,224,224)
        input_obs = torch.tensor(img)
        #        print(input_obs.shape)
        input_obs = input_obs.to(device)
        input_obs = input_obs.view(1, 3, 224, 224)
        out = net(input_obs.float())
        print(out)
        out_result = out.detach().cpu().numpy()
        # _, predicted = torch.max(outputs.data, 1)
        # correct += (predicted == labels).sum().item()

        # print(losssum)

        steering = -out_result[0][0]
        #       print("--caculate time--")
        #       print(time.time()-time0)
        # _, predicted = torch.max(out.data, 1)
        # print("out:"+str(acts_space[predicted.item()]))
        # index_act = predicted.item()
        # out = model()

        #      print('total:'+str(acts_space[index_act]))
        #       print('steering:'+str(steering))
        #        print('throttle:'+str(throttle))
        # throttle = (out[0][0][0].numpy())
        # steering = (out[1][0][0].numpy())
        # print(out)
        throttle = 0.35

        print("==Throttle:" + str(throttle) + "===Steering:" + str(steering) + "  ================")

        if throttle > 0:
            # throttle = min(throttle,1)
            throttle_pulse = dk.utils.map_range(throttle, 0, 1, THROTTLE_STOPPED_PWM, THROTTLE_FORWARD_PWM)
        else:
            # throttle = max(throttle,-1)
            throttle_pulse = dk.utils.map_range(throttle, -1, 0,
                                                THROTTLE_REVERSE_PWM, THROTTLE_STOPPED_PWM)

        # if steering > 0:
        # steering = min(steering,1)
        # else:
        # sterring = max(steering,-1)
        steering_pulse = dk.utils.map_range(steering,
                                            -1, 1,
                                            STEERING_LEFT_PWM,
                                            STEERING_RIGHT_PWM)

        pwm.set_pwm(THROTTLE_CHANNEL, 0, int(throttle_pulse))
        #     print("throttle",throttle_pulse)
        # time.sleep(0.1)
        # print("=================")
        pwm.set_pwm(STEERING_CHANNEL, 0, int(steering_pulse))
        #         print("steering", steering_pulse)
        # time.sleep(0.1)
        #         print("=================")

        # cur_out = []
        # cur_out.append(out[0].numpy())
        # cur_out.append(out[1].numpy())
        # outputs.append(out)
        # print(out)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
        # count += 1
        #        print("total time")
        #        print(time.time()-time0)

    pwm.set_pwm(THROTTLE_CHANNEL, 0, int(THROTTLE_STOPPED_PWM))
    time2 = time.time()
    # np.save("outputs.npy",outputs)
    # print('time overall: ', time2 - time1)
    # print('average time: ', (time2 - time1) / 200)





