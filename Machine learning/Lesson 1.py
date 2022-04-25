import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train)                          # training sample size
w = [0, -1]                                     # the initial value of the vector w
a = lambda x: np.sign(x[0]*w[0] + x[1]*w[1])    # decision rule
N = 50                                          # maximum number of iterations
L = 0.1                                         # weight change step
e = 0.1                                         # a small addition for w0 so that there is a gap between the dividing line and the boundary image

last_error_index = -1                           # last error index

for n in range(N):
    for i in range(n_train):
        if y_train[i]*a(x_train[i])<0:          # search by observation
            w[0] = w[0] + L * y_train[i]        # if the classification error
            last_error_index = i                # then the weight adjustment w0

    Q = sum([1 for i in range(n_train) if y_train[i]*a(x_train[i])<0])
    if Q == 0:                   # classification quality indicator (number of errors)
        break                    # stop if everything is correctly classified
if last_error_index > -1:
    w[0] = w[0] + e * y_train[last_error_index]
    
print(w)

line_x = list(range(max(x_train[:, 0])))       # forming a dividing line graph
line_y = [w[0]*x for x in line_x]

x_0 = x_train[y_train == 1]                        # formation of points for the 1st
x_1 = x_train[y_train == -1]                       # and 2nd grade

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel('length')
plt.xlabel('width')
plt.grid(True)
plt.show()






