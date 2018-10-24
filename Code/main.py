from architecture import Classifier
from utils import one_hot_encoder
from utils import one_hot_decoder


if __name__ == '__main__':
    c1 = Classifier()
    c1.train_model()
    c1.predict_model()
