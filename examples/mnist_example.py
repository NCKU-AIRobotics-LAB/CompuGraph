import gzip
import numpy as np
import os
from urllib.request import urlretrieve
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
from compugraph import mlp1, mlp2
import matplotlib.pyplot as plt

result_path = os.path.join(os.path.dirname(__file__), 'results')

# train-images-idx3-ubyte.gz: training set images (9912422 bytes)
# train-labels-idx1-ubyte.gz: training set labels (28881 bytes)
# t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)
# t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

def load_idx(filepath):
    with gzip.open(filepath, 'rb') as fin:
        idx_data = fin.read()
        idx = 0
        unsigned = idx_data[idx + 2] == 8
        dim = idx_data[idx + 3]
        idx += 4

        dim_shape = []
        for i in range(dim):
            dim_shape.append(int.from_bytes(idx_data[idx:idx+4], 'big', signed=False))
            idx += 4

        if unsigned:
            return np.frombuffer(idx_data[idx:], np.uint8).reshape(dim_shape)
        return None

def select_images():
    if (not os.path.exists('data')):
        os.makedirs('data')
    urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'data/train-images-idx3-ubyte.gz')
    urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
    urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'data/t10k-images-idx3-ubyte.gz')
    urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz')
    train_images_data = load_idx('data/train-images-idx3-ubyte.gz')
    train_labels_data = load_idx('data/train-labels-idx1-ubyte.gz')
    test_images_data = load_idx('data/t10k-images-idx3-ubyte.gz')
    test_labels_data = load_idx('data/t10k-labels-idx1-ubyte.gz')

    images_list = []
    labels_list = []
    for n in range(10):
        idx = np.where(train_labels_data == n)[0][:]
        images_list.append(train_images_data[idx])
        labels_list.extend(train_labels_data[idx])
    train_images = np.vstack(images_list)
    train_labels = np.vstack(labels_list).reshape(-1)
    train_labels = np.eye(10)[train_labels]

    images_list = []
    labels_list = []
    for n in range(10):
        idx = np.where(test_labels_data == n)[0][:]
        images_list.append(test_images_data[idx])
        labels_list.extend(test_labels_data[idx])
    test_images = np.vstack(images_list)
    test_labels = np.vstack(labels_list).reshape(-1)
    test_labels = np.eye(10)[test_labels]

    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = select_images()

X_train = (train_images.reshape(train_images.shape[0], -1) / 255.0).astype(np.float64)
Y_train = train_labels.astype(np.float32)
X_test = (test_images.reshape(test_images.shape[0], -1) / 255.0).astype(np.float64)
Y_test = test_labels.astype(np.float64)

if (not os.path.exists(result_path)):
    os.makedirs(result_path)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

print('\nMLP1:\n')

results = mlp1(X_train, Y_train, X_test, Y_test, 10)

x = list(range(1, len(results)))
train_err = [1 - results[i]['accuracy'] for i in range(0, len(results) - 1)]
val_err = [1 - results[i]['val_accuracy'] for i in range(0, len(results) - 1)]

plt.figure(1)
plt.plot(x, train_err)
plt.xlabel('epoch')
plt.ylabel('Training Error')
plt.savefig(os.path.join(result_path, 'train_1.png'))

plt.figure(2)
plt.plot(x, val_err)
plt.xlabel('epoch')
plt.ylabel('Validation Error')
plt.savefig(os.path.join(result_path, 'test_1.png'))


print('\nMLP2:\n')

results = mlp2(X_train, Y_train, X_test, Y_test, 10)

x = list(range(1, len(results)))
train_err = [1 - results[i]['accuracy'] for i in range(0, len(results) - 1)]
val_err = [1 - results[i]['val_accuracy'] for i in range(0, len(results) - 1)]

plt.figure(3)
plt.plot(x, train_err)
plt.xlabel('epoch')
plt.ylabel('Training Error')
plt.savefig(os.path.join(result_path, 'train_2.png'))

plt.figure(4)
plt.plot(x, val_err)
plt.xlabel('epoch')
plt.ylabel('Testing Error')
plt.savefig(os.path.join(result_path, 'test_2.png'))