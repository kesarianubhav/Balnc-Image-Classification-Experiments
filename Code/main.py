from architecture import Classifier
from utils import one_hot_encoder
from utils import one_hot_decoder


def load_image_dir()
    return


def img_to_tensor(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    tensor = img_to_array(img)
    # print(tensor.shape)
    tensor = np.expand_dims(tensor, axis=0)
    # tensor = preprocess_input(tensor)
    print("Image """ + str(image_path) +
          " "" converted to tensor with shape " + str(tensor.shape))
    return tensor


if __name__ == '__main__':
    c1 = Classifier()
    load_image_dir('Datasets')
    # X_
    c1.train_model()
    c1.predict_model()
