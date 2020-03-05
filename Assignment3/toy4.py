import numpy 
import matplotlib.pyplot as plt 

data = [12, 15, 40,  20, 41, 29,  8,   51, 13, 20, 
        14, 24, 13,  15, 16, 25,  145, 12, 30, 15, 
        13, 26, 12,  18, 31, 16,  14,  18, 14, 10, 
        24, 8,  13,  10, 26, 13,  21,  17, 43, 22, 
        20, 31, 18,  21, 19, 21,  16,  23, 21, 18, 
        16, 64, 54,  13, 17, 14,  22,  41, 52, 17, 
        25, 54, 15,  27, 28, 17,  30,  80, 12, 22, 
        21, 52, 20,  64, 18, 15,  26,  25, 30, 98, 
        19, 49, 16,  38, 20, 55,  179, 14, 21, 13, 
        18, 15, 28,  14, 15, 131, 57,  39, 31, 18, 
        16, 17, 108, 38, 20, 20,  64,  42, 24, 29, 
        17]

cnt = [0.] * (max(data))
for num in data:
    cnt[num - 1] += 1

plt.scatter([i for i in range(max(data))], cnt, s = 5)
plt.title('Distribution of the star size')
plt.xlabel('Star Size')
plt.ylabel('Frequency')
plt.savefig('../eee.png', dpi = 400)