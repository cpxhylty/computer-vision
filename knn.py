import lbp
import load_dataset as ld
import numpy as np

test_number = 100
train_number = 40000 # no greater than 40000
K = 11
correction_number = 0
x_train, y_train, file_names = ld.load_CIFAR_batch("./cifar-100-python/train")
x_test, y_test = x_train[40000:40000+test_number].reshape(test_number, 32*32*3) / 255, y_train[40000:40000+test_number]
x_train, y_train = x_train[0:train_number].reshape(train_number, 32*32*3) / 255, y_train[0:train_number]
y_hats_over_examples = []


# unitization
for i in range(train_number):
    x_train[i] /= np.linalg.norm(x_train[i])
for i in range(test_number):
    x_test[i] /= np.linalg.norm(x_test[i])

for i in range(test_number):
    image_test = x_test[i]
    distances = [10000] * K
    y_hats = [1234] * K
    for j in range(train_number):
        image_train = x_train[j]
        temp_distance = np.linalg.norm(image_test - image_train)
        for k in range(K):
            if temp_distance < distances[k]:
                distances.insert(k, temp_distance)
                y_hats.insert(k, y_train[j])
                break
        '''if temp_distance < distance:
            y_hat = y_train[j]
            distance = temp_distance'''
    y_hats = y_hats[0:K]
    distances = distances[0:K]
    number_of_y_hat = 0
    y_hat = 1234
    record = {}
    for k in range(K):
        record[y_hats[k]] = record.get(y_hats[k], 0) + 1
        if record[y_hats[k]] > number_of_y_hat:
            number_of_y_hat = record[y_hats[k]]
            y_hat = y_hats[k]
    if y_hat == y_test[i]:
        correction_number += 1
    y_hats_over_examples.append(y_hat)
print('K: ' + str(K))
print('test examples: ' + str(test_number))
print('train examples: ' + str(train_number))
print(y_test[0:test_number])
print(np.asarray(y_hats_over_examples))
print('accuracy: ' + str(correction_number / test_number))


