
from Adafruit_GPIO import I2C

import torch.nn as nn

import time
import cv2
import torch

import Adafruit_PCA9685
import donkeycar as dk


# tf.debugging.set_log_device_placement(
#    True
# )

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
# pwm.set_pulse(STEERING_CHANNEL,0,int(THROTTLE_STOPPED_PWM))

if __name__ == '__main__':

    device = 'cuda'
    print(device)
    acts_space = [(-1, 0.05), (-1, 0.15000000000000002), (-1, 0.25), (-1, 0.35000000000000003), (-1, 0.45), (-1, 0.55),
                  (-1, 0.6500000000000001), (-1, 0.7500000000000001), (-1, 0.8500000000000001),
                  (-1, 0.9500000000000001), (0, 0.05),
                  (0, 0.15000000000000002), (0, 0.25), (0, 0.35000000000000003), (0, 0.45), (0, 0.55),
                  (0, 0.6500000000000001),
                  (0, 0.7500000000000001), (0, 0.8500000000000001), (0, 0.9500000000000001), (1, 0.05),
                  (1, 0.15000000000000002),
                  (1, 0.25), (1, 0.35000000000000003), (1, 0.45), (1, 0.55), (1, 0.6500000000000001),
                  (1, 0.7500000000000001),
                  (1, 0.8500000000000001), (1, 0.9500000000000001)]

    ########load model
    best_filename = 'model2_newbest.tar'


    def gstreamer_pipeline(capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21,
                           flip_method=0):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
            capture_width, capture_height, framerate, flip_method, output_width, output_height)


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
    # camera = cv2.VideoCapture(0)

    ret = True
    state = torch.load(best_filename)

    net = Net()
    net.load_state_dict(state['state_dict'])
    net.to(device)
    # print(model.summary())
    time1 = time.time()
    outputs = []
    count = 0
    net.eval()

    while count < 800:
        count = count + 1

        time0 = time.time()
        ret, img = camera.read()
        cv2.imwrite("./test_img/"+str(count)+".jpg",img)
        print(ret)
        img=img.reshape(3,224,224) 
        input_obs = torch.tensor(img)
        #        print(input_obs.shape)
        input_obs = input_obs.to(device)
        input_obs = input_obs.view(1, 3, 224, 224)
        print(input_obs)
        out = net(input_obs.float())
        print(out)
        y=out
        print('y:       ',y)
        print('type(y): ',type(y))
        print('y.dtype: ',y.dtype)  # y的具体类型
        out_result = out.detach().cpu().numpy()
        print(out_result)
        # throttle = out_result[0, 0]
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

        print("==Throttle:"+str(throttle)+"===Steering:"+str(steering)+"  ================")

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
    print('time overall: ', time2 - time1)
    # print('average time: ', (time2 - time1) / 200)



