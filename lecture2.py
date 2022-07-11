import torch
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklean import datasets

# lectur 2 
digits_data = datasets.load_digits()

n_imp = 10
plt.figure(figsize(10,4))
for i in range(n_imp):
    ax = plt.subplot(2,5,i+1)
    plt.imshow(digits_data.data[i].reshape(8,8)),cmap="Greys_r")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



