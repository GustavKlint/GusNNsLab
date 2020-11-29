import numpy as np
import matplotlib.pyplot as plt

# Here we define the ReLU function.
def relu(x):
    """ReLU returns 1 if x>0, else 0."""
    return np.maximum(0,x)

def reludx(x):
    return np.where(x <= 0, 0, 1)

def reludx2(x):
    return np.where(x <= -4, 0, 1)

def reludx3(x):
    return np.where(x <= -4, 1, 1)

# Data for ReLU
x1 = np.linspace(-10,10,21)
y1 = relu(x1)

# Data for ReLU derivative
x2 = np.linspace(-10,10,10000000)
y2 = reludx(x2)

# Data for shifted ReLU derivative
x3 = np.linspace(-10,10,100000)
y3 = reludx2(x3)

# Data for shifted ReLU derivative
x4 = np.linspace(-10,10,100000)
y4 = reludx3(x4)

# Figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([-10,10])
plt.title("ReLU Activation Function")

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x1, y1)

# Figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([-2,2])
ax.set_xlim([-5,5])
plt.title("Correct ReLU Derivative")

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x2, y2)

# Figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([-2,2])
ax.set_xlim([-5,5])
plt.title("Shifted FakeReLU Derivative")

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x3, y3)

# Figure
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim([-2,2])
ax.set_xlim([-5,5])
plt.title("FakeReLU Derivative")

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.plot(x4, y4)
plt.show()
