import lbp
import load_dataset as ld
import numpy as np

test_number = 100
train_number = 1000 # no greater than 40000
K = 1
correction_number = 0
x_train, y_train, file_names = ld.load_CIFAR_batch("./cifar-100-python/train")
x_test, y_test = x_train[40000:40000+test_number].reshape(test_number, 32*32*3) / 255, y_train[40000:40000+test_number]
x_train, y_train = x_train[0:train_number].reshape(train_number, 32*32*3) / 255, y_train[0:train_number]
y_hats = []


'''# unitization
for i in range(train_number):
    x_train[i] /= np.linalg.norm(x_train[i])
for i in range(test_number):
    x_test[i] /= np.linalg.norm(x_test[i])'''

for i in range(test_number):
    image_test = x_test[i]
    distance = 10000
    y_hat = 1234
    for j in range(train_number):
        image_train = x_train[j]
        temp_distance = np.linalg.norm(image_test - image_train)
        if temp_distance < distance:
            y_hat = y_train[j]
            distance = temp_distance
    if y_hat == y_test[i]:
        correction_number += 1
    y_hats.append(y_hat)
print('test examples: ' + str(test_number))
print('train examples: ' + str(train_number))
print(y_test[0:test_number])
print(np.asarray(y_hats))
print('accuracy: ' + str(correction_number / test_number))


