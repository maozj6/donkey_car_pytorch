# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import model
import torch
import time  # 引入time模块

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    steering=[-1,0,1]

    gas=[]
    for i in range(10):
        gas.append(0.05+(i)*0.1)
    action_space = []
    for i in range(3):
        for j in range(10):
            action_space.append((steering[i],gas[j]))

    net=model.Net()
    all=[]
    imgae=torch.rand((1,224,224), out=None)
    for i in range(1000):
        all.append(imgae)
    time1=time.time()
    for i in range(1000):
        out=net(imgae)
        print(time.time()-time1)

    print_hi('PyCharm')
    print(action_space)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
