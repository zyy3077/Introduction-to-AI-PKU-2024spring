import numpy as np
import matplotlib.pyplot as plt
x=np.array([0,1,2,3,4,5,6,7,8,9,10])
y_sin=np.sin(x*36*np.pi/180)
y_f=x**2-10*x+25
img=plt.imread('mnist.jpg')
fig=plt.figure(figsize=(12,4))
fig.patch.set_facecolor('lightgrey')
plt.subplot(131)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.plot(x,y_sin)
plt.subplot(132)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.bar(x,y_f)
plt.subplot(133)
plt.axis('off')
plt.imshow(img,cmap='gray')
plt.show()