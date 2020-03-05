import numpy as np
import random
import matplotlib.pyplot as plt


Xin = [[0.5, 0.5, 1],
       [0.5, 479.5, 1],
       [719.5, 0.5, 1],
       [719.5, 479.5, 1]]

Xout = [[0.5, 0.5],
        [700.5, 479.5],
        [2100.5, 0.5],
        [1400.5, 479.5]]

Xin = np.asarray(Xin)
Xout = np.asarray(Xout)

Xout = np.matmul(np.transpose(Xin), Xout)
Xin = np.matmul(np.transpose(Xin), Xin)

Xfinal = np.matmul(np.linalg.inv(Xin), Xout)

# print(Xfinal)
# print(np.matmul(np.asarray([0.5, 0.5, 1]), Xfinal))
# print(np.matmul(np.asarray([0.5, 479.5, 1]), Xfinal))
# print(np.matmul(np.asarray([719.5, 0.5, 1]), Xfinal))
# print(np.matmul(np.asarray([719.5, 479.5, 1]), Xfinal))



# This is the projective transformation method
# Code reference
# https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript/339033#339033

# 356.5 331.5 356.5 372.5 
# 328.5 331.5 313.5 271.5 
# 1 1 1 1 


# 357.5 332.5 357.5 375.5 
# 178.5 180.5 163.5 120.5 
# 1 1 1 1 
# A = np.asarray([[356.5, 331.5, 356.5], [328.5, 331.5, 313.5], [1, 1, 1]])
# B = np.asarray([[357.5, 332.5, 357.5], [178.5, 180.5, 163.5], [1, 1, 1]])
# A_x = np.asarray([[372.5], [271.5], [1]])
# B_x = np.asarray([[375.5], [120.5], [1]])


A = np.asarray([[0.5, 0.5, 719.5], [0.5, 479.5, 0.5], [1, 1, 1]])
B = np.asarray([[0.5, 350.5, 1050.5], [0.5, 479.5, 0.5], [1, 1, 1]])
A_x = np.asarray([[719.5], [479.5], [1]])
B_x = np.asarray([[700.5], [479.5], [1]])


Scaling_Col_Scr = np.matmul(np.linalg.inv(A), A_x)
Scaling_Col_Desti = np.matmul(np.linalg.inv(B), B_x)

A = np.multiply(A, np.transpose(Scaling_Col_Scr))
B = np.multiply(B, np.transpose(Scaling_Col_Desti))


C1 = np.matmul(B, np.linalg.inv(A))
C2 = np.matmul(A, np.linalg.inv(B))
print(C1, "\n", C2)

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
        temp = np.matmul(C1, np.asarray([[x], [y], [1]]))
        X_after.append(temp[0] / temp[2])
        Y_after.append(temp[1] / temp[2])

        temp = np.matmul(C2, np.asarray([[temp[0] / temp[2]], [temp[1] / temp[2]], [1]]))
        X_rev.append(temp[0] / temp[2])
        Y_rev.append(temp[1] / temp[2])


plt.scatter(Y_ori, X_ori, c = 'red', s = 0.5)
plt.scatter(Y_after, X_after, c = 'blue', s = 0.5)
# plt.xlim((-1000, 1000))
# plt.ylim((-1000, 1000))
plt.savefig("../Mapping_rect_TO_Trapezoid.png", dpi = 400)
plt.clf()
plt.scatter(Y_rev, X_rev, c = 'green', s = 0.5)
plt.scatter(Y_after, X_after, c = 'blue', s = 0.5)
# plt.xlim((-1000, 1000))
# plt.ylim((-1000, 1000))
plt.savefig("../Mapping_Trapezoid_TO_rect.png", dpi = 400)

