import mxnet as mx
import numpy as np
import pandas as pd


def record_loader(record_file):
    record = mx.recordio.MXRecordIO(record_file, 'r')
    images_in_record = []
    while True:
        item = record.read()
        header, img = mx.recordio.unpack_img(item)
        images_in_record.append(img)
        print("Total Images in Record: " + str(len(images_in_record)))
        if not item:
            break
        print("image:\n" + str(img))
    # print("Total Images in Record: " + str(len(images_in_record)))
    record.close()
    return images_in_record


if __name__ == '__main__':
    record_loader('example_rec.rec')
