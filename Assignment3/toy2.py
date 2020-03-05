import numpy as np
import random
import matplotlib.pyplot as plt



# Xin = [[0, 0, 1], [0, 100, 1], [100, 100, 1], [100, 0, 1],
#        [0, 50, 1], [100, 50, 1],
#        [50, 50, 1],
#        [50, 100, 1], [50, 0, 1]]

# Xout = [[0, 0, 1], [80, 100, 1], [120, 100, 1], [200, 0, 1],
#         [40, 50, 1], [160, 50, 1],
#         [100, 50, 1],
#         [100, 100, 1], [100, 0, 1]]

Xin  = [
        [356.5,328.5,1],
        [331.5,331.5,1],
        # [356.5,313.5,1],
        [372.5,271.5,1],
        [321.5,376.5,1]]

Xout = [
        [357.5,178.5,1],
        [332.5,180.5,1],
        # [357.5,163.5,1],
        [375.5,120.5,1],
        [323.5,221.5,1]]
# Xin = [
#         [358,310,1],
#         [360,333,1],
#         [606,416,1],
#         [204,243,1]]
# Xout =  [
#         [360,160,1],
#         [360,183,1],
#         [578,266,1],
#         [192,84,1]]



# Calculate the X part
U, V = [], []
for pnt in range(len(Xin)):
    U.append(Xin[pnt] + [0, 0, 0] + [-Xin[pnt][0] * Xout[pnt][0], -Xin[pnt][1] * Xout[pnt][0]])
    U.append([0, 0, 0] + Xin[pnt] + [-Xin[pnt][0] * Xout[pnt][1], -Xin[pnt][1] * Xout[pnt][1]])
    V.append(Xout[pnt][0])
    V.append(Xout[pnt][1])


U = np.asarray(U)
V = np.asarray(V)

print(U, "\n", V, '\n')

# U = np.matmul(np.linalg.inv(np.matmul(U.T, U)), U.T)
U = np.linalg.pinv(U)
print(U)

T = np.matmul(U, V).tolist() + [1]
T = np.asarray(T)
T = T.reshape(3, 3).tolist()

for row in T:
    print(row)

T_inv = np.linalg.inv(T)

for row in T_inv:
    print(row)



X_ori = []
Y_ori = []
X_after = []
Y_after = []
X_rev = []
Y_rev = []

for _ in range(10000):
        x, y = 720 * random.random(), 480 * random.random()
        X_ori.append(x)
        Y_ori.append(y)
        temp = np.matmul(T, np.asarray([[x], [y], [1]]))
        X_after.append(temp[0] / temp[2])
        Y_after.append(temp[1] / temp[2])

        temp = np.matmul(T_inv, np.asarray([[temp[0] / temp[2]], [temp[1] / temp[2]], [1]]))
        X_rev.append(temp[0] / temp[2])
        Y_rev.append(temp[1] / temp[2])

plt.scatter(Y_ori, X_ori, c = 'red', s = 0.1)
plt.scatter(Y_after, X_after, c = 'blue', s = 0.1)
plt.xlim((-1000, 1000))
plt.ylim((-1000, 1000))
plt.savefig("../Mapping_rect_TO_Trapezoid.png", dpi = 400)
plt.clf()
plt.scatter(Y_rev, X_rev, c = 'green', s = 0.1)
plt.scatter(Y_after, X_after, c = 'blue', s = 0.1)
plt.xlim((-1000, 1000))
plt.ylim((-1000, 1000))
plt.savefig("../Mapping_Trapezoid_TO_rect.png", dpi = 400)

