import numpy as np
import torch
import mogptk
def low_fidelity(x):
    if np.abs(x)<=1:
        return 1
    else:
        return 0

def high_fidelity(x):
    if x<-2 or x>2:
        return 0
    if x>-2 and x<0:
        return 2+x
    if x>0 and x<2:
        return 2-x




def generate_points(start,stop,num_points,subsequence):
    eps = 0.5
    len = stop-start

    step = eps * (1 / len)
    points_x = []

    for i in range(num_points):
            start = np.random.uniform(start, stop, 1)
            x=np.linspace(start, subsequence * step + start, subsequence)
            for i in x:
                points_x.append(i[0])
    return np.asarray(points_x)


x=generate_points(-15,15,300,6)


points1=[low_fidelity(i) for i in x ]
points2=[high_fidelity(i) for i in x]

f2=[]
f1=[]
i=0
while i< len(points2):
    f2.append(points2[i:i+6])
    f1.append(points1[i:i+6])
    i+=6


points1 = [low_fidelity(i) for i in x]
points2 = [high_fidelity(i) for i in x]

