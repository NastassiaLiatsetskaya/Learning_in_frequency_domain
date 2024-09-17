from scipy.fft import rfft
import numpy as np
d=np.asarray([1,2,3,4,5,6,7,8,9,10,11,12])
transform=rfft(d)
print(transform)
reals=[i.real for i in transform]
print(reals)
imag=[i.imag for i in transform]
print(imag)

w=np.asarray([[1,2,3],[4,5,6],[7,8,9]])
print(w[:,1:])