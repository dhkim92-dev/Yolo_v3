import numpy as np
from model import Yolo
from utils import load_darknet_weights
from absl import logging

CLASSES = 80
WEIGHTS = './data/yolo.weights'
OUTPUT = './checkpoints/yolov3.tf'

def main():
    yolo = Yolo(classes=CLASSES)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, WEIGHTS)
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(OUTPUT)
    logging.info('weights saved')

if __name__ == '__main__':
    main()