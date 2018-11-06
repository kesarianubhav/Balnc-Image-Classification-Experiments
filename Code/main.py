import numpy as np
import pandas as pd
import keras.backend as K
import keras
# from keras.models import K
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics import accuracy_score, f1_score

from utils import getFilesInDir, create_labels
# from rnn_backend import RnnClassifier, preprocess_input_rnn
from architecture import Classifier
from utils import one_hot_to_integer
from utils import one_hot
from utils import missclassification_rate
from utils import test_train_split

# For Matching Images
import imagehash as ihash

fraction = 100


def img_to_tensor(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    tensor = img_to_array(img)
    # print(tensor.shape)
    tensor = np.expand_dims(tensor, axis=0)
    # tensor = preprocess_input(tensor)
    print("Image """ + str(image_path) +
          " "" converted to tensor with shape " + str(tensor.shape))
    return tensor


def input_maker(input_folder, target_size, output_classes):
    images = getFilesInDir(input_folder)
    print("File Map : {}".format(images.keys()))
    # images = getFilesInDir(folder)
    images_train, images_test = test_train_split(images, fraction)
    tensors_train = []
    tensors_test = []
    for i, j in images_train.items():
        for k in j:
            tensors_train.append(img_to_tensor(k, target_size=(target_size)))
            print("Train Tensor :" + str(k) + " Created..\n")
    for i, j in images_test.items():
        for k in j:
            tensors_test.append(img_to_tensor(k, target_size=(target_size)))
            print("Test Tensor :" + str(k) + " Created..\n")

    print("Total Training Tensors:" + str(len(tensors_train)) +
          " each of shape " + str(tensors_train[0].shape))
    print("Total Testing Tensors:" + str(len(tensors_train)) +
          " each of shape " + str((tensors_test[0].shape)))
    index = 0
#     for i,j in labels_train:
    labels_train = create_labels(images_train, output_classes=output_classes)
    labels_test = create_labels(images_test, output_classes=output_classes)

    print("Total Training Lables created:" + str(len(labels_train)))
    print("Total Testing Lables created:" + str(len(labels_test)))

    return (tensors_train, labels_train, tensors_test, labels_test)


def get_best_matches(input, class_label, num_of_matches=3, folder='Sport', target_size=(224, 224)):
    # tensors = []
    indexes = []
    hashes = []
    matches = []
    input_hash = ihash.average_hash(input)
    file_map = getFilesInDir(folder)[class_label]
    for i in file_map:
        tensor = (img_to_tensor(i, target_size=(target_size)))
        hashes.append(ihash.average_hash(tensor) - input_hash)
    # for i in file_map:
    indexes = [i for i in range(1, len(hashes) + 1)]
    s = sorted(zip(hashes, indexes))
    results = [file_map[i]
               for i in range(len(hashes) - 1, len(hashes) - num_of_matches, -1)]
    return results


def normalizer(image):
    image = (image - np.mean(image)) / 255
    return image


if __name__ == '__main__':
    c1 = Classifier()
    X_train, Y_train, X_test, Y_test = input_maker(
        'Sport', (224, 224), 4)

    print(X_train[:4])
#     print((Y_train[:200]))
    ls = [one_hot_to_integer(i) for i in Y_train]
    print(set(ls))
#     sys.exit(0)

    X_train = normalizer(X_train)
    X_test = normalizer(X_test)

    from sklearn.utils import shuffle
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test = shuffle(X_test, Y_test, random_state=0)

    c1.create_architecture(input_shape=(224, 224, 3), output_dimension=4)
    c1.train_model(X_train, Y_train)

    predicted_labels = c1.predict(X_test)
    predicted = [i[0] for i in predicted_labels]
    print(predicted)
    actual = [one_hot_to_integer(i) for i in Y_test]
    print(actual)
    print("Missclassification Rate: {}".format(
        missclassification_rate(predicted, actual)))

    # image_name = input('enter Image : to get best similar Image')
