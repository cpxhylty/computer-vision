import pickle as p
import numpy as np
from PIL import Image
import edge_filter as ef
import matplotlib.pyplot as plt


def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        x = datadict[b'data']
        #y_coarse = datadict[b'coarse_labels']
        y = datadict[b'fine_labels']
        file_names = datadict[b'filenames']
        #batch_labels = datadict[b'batch_label']
        x = x.reshape(50000, 3, 32, 32)
        y = np.array(y)
        return x, y, file_names
    # dict_keys([b'data', b'coarse_labels', b'fine_labels', b'filenames', b'batch_label'])

def visualize(image):
    img0 = image[0]
    img1 = image[1]
    img2 = image[2]
    i0 = Image.fromarray(img0)
    i1 = Image.fromarray(img1)
    i2 = Image.fromarray(img2)
    img = Image.merge("RGB", (i0, i1, i2))
    #img.show()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    return

'''
# use cifar image
x_train, y_train, file_names = load_CIFAR_batch("./cifar-100-python/train")
order = 20
visualize(x_train[order])
#print(y_train[order])
#print(file_names[order])
ef.edge_filter(x_train[order], stride=1)
'''

'''
# use user image
user_image = np.array(Image.open(''))
ef.edge_filter(user_image, explicit_factor=2)
'''