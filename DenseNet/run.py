import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]

import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions

import MyDenseNet

if __name__ == "__main__":
    img = cv2.imread('../img/cat.jpg', )
    img = cv2.resize(img, (224, 224))
    print img.shape
    batch = np.reshape(img, [1, 224, 224, 3])
    batch = MyDenseNet.preprocess(batch)

    model = MyDenseNet.create_model(224)
    predictions = model.predict_proba(batch)
    #print predictions
    label = decode_predictions(predictions)
    print label

    pass

