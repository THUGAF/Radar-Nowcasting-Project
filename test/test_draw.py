import os
import matplotlib.pyplot as plt
import numpy as np


L1_train_loss = np.loadtxt("C:/Users/GAF/Desktop/all_results/results_1L_S/loss/train_loss.txt")
L1_val_loss = np.loadtxt("C:/Users/GAF/Desktop/all_results/results_1L_S/loss/val_loss.txt")
L3_train_loss = np.loadtxt("C:/Users/GAF/Desktop/all_results/results_3L_S/loss/train_loss.txt")
L3_val_loss = np.loadtxt("C:/Users/GAF/Desktop/all_results/results_3L_S/loss/val_loss.txt")


fig = plt.figure(figsize=(8, 6), dpi=150)
ax = plt.subplot(111)
ax.plot(range(1, len(L1_train_loss) + 1), L1_train_loss, 'b')
ax.plot(range(1, len(L1_val_loss) + 1), L1_val_loss, 'r')
ax.plot(range(1, len(L3_train_loss) + 1), L3_train_loss, 'b--')
ax.plot(range(1, len(L3_val_loss) + 1), L3_val_loss, 'r--')
ax.set_xlabel('epoch')
ax.set_ylim([0, 0.3])
ax.legend(['L1 train loss', 'L1 val loss', 'L3 train loss', 'L3 val loss'])

plt.savefig('loss_S' + '.png')
plt.savefig('loss_S' + '.eps', format='eps')
plt.close()
