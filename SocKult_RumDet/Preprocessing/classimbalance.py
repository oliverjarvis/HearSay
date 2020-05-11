import numpy as np
y_train =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\train\\labels.npy")

u, c1 = np.unique(y_train, return_counts = True)

print(c1)

import numpy as np
y_train =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\test\\labels.npy")

u, c2 = np.unique(y_train, return_counts = True)

print(c2)

import numpy as np
y_train =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\labels.npy")

u, c3 = np.unique(y_train, return_counts = True)

print(c3)