

import json

import cv2
import numpy as np

if __name__ == '__main__':
    acts_space=[(-1, 0.05), (-1, 0.15000000000000002), (-1, 0.25), (-1, 0.35000000000000003), (-1, 0.45), (-1, 0.55),
     (-1, 0.6500000000000001), (-1, 0.7500000000000001), (-1, 0.8500000000000001), (-1, 0.9500000000000001), (0, 0.05),
     (0, 0.15000000000000002), (0, 0.25), (0, 0.35000000000000003), (0, 0.45), (0, 0.55), (0, 0.6500000000000001),
     (0, 0.7500000000000001), (0, 0.8500000000000001), (0, 0.9500000000000001), (1, 0.05), (1, 0.15000000000000002),
     (1, 0.25), (1, 0.35000000000000003), (1, 0.45), (1, 0.55), (1, 0.6500000000000001), (1, 0.7500000000000001),
     (1, 0.8500000000000001), (1, 0.9500000000000001)]
    obs = []
    gas = []
    steering = []
    act_sp = []
    clas = []
    # "/home/mao/23Spring/cars/donkey_train/data"
    f = open("data/catalog_10.catalog")  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        # print(line)
        user_dict = json.loads(line)
        single_angle = user_dict['user/angle']
        steering.append(int(single_angle))
        single_gas = user_dict['user/throttle']
        if single_angle == -1:
            jnedx = 0
        elif single_angle == 0:
            jnedx = 1
        elif single_angle == 1:
            jnedx = 2
        else:
            jnedx = 3
        index = int(single_gas * 10)
        # print(index)
        act_sp.append(jnedx * 10 + index)

        gas.append(single_gas)
        img_path = user_dict['cam/image_array']

        observation = cv2.imread("/home/mao/23Spring/cars/donkey_train/data/images/" + img_path)
        grey_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        obs.append(grey_obs)
        # gas.append()
        # print(line, end = '')　      # 在 Python 3 中使用
        line = f.readline()

    f.close()
    total_obs=np.array(obs)
    total_act=np.array(act_sp)

    # np.savez_compressed("test.npz", obs=obs, lbl=act_sp)

    for i in range(11,13):
        print(i)
        obs=[]
        gas=[]
        steering=[]
        act_sp=[]
        clas=[]
        # "/home/mao/23Spring/cars/donkey_train/data"
        f = open("data/catalog_"+str(i)+".catalog")  # 返回一个文件对象
        line = f.readline()  # 调用文件的 readline()方法
        while line:
            # print(line)
            user_dict = json.loads(line)
            single_angle=user_dict['user/angle']
            steering.append(int(single_angle))
            single_gas=user_dict['user/throttle']
            if single_angle==-1:
                jnedx=0
            elif single_angle==0:
                jnedx=1
            elif single_angle==1:
                jnedx=2
            else:
                jnedx=3
            index=int(single_gas*10)
            # print(index)
            act_sp.append(jnedx*10+index)


            gas.append(single_gas)
            img_path=user_dict['cam/image_array']

            observation=cv2.imread("/home/mao/23Spring/cars/donkey_train/data/images/"+img_path)
            grey_obs=cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            obs.append(grey_obs)
            # gas.append()
            # print(line, end = '')　      # 在 Python 3 中使用
            line = f.readline()

        f.close()
        temp_obs = np.array(obs)
        temp_act = np.array(act_sp)
        total_obs=np.append(total_obs,temp_obs,axis=0)
        total_act=np.append(total_act,temp_act,axis=0)
    np.savez_compressed("test.npz",obs=total_obs,lbl=total_act)
