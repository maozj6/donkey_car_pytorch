
import numpy as np
if __name__ == '__main__':
    train=np.load("train.npz")

    obs=train["obs"]
    acts=train["lbl"]

    print("end")
