import cv2
from model import Yolo
import time
import tensorflow as tf
import numpy as np

MS_COCO_CLASSES = 80
DEFAULT_SIZE = 416

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img

@tf.function
def resizeImage(image, size) :
	image = tf.image.resize(image, (size,size))
	image = image / 255

	return image
	


def main() :
	#print('model loading...')
	yolo = Yolo(classes=MS_COCO_CLASSES) ## MS COCO 
	#print('model loaded')
	cam = cv2.VideoCapture(0)

	#print('Yolo weights loading...')
	yolo.load_weights('./checkpoints/yolov3.tf')
	#print('weight loaded')

	#print('classes name loading...')
	class_name = [c.strip() for c in open('./data/coco.names').readlines()]
	#print('classes loaded')

	if not cam.isOpened() :
		print('camera is not opened')
		exit()

	while True :
		_,frame = cam.read()

		if frame is None :
			print('failed to read frame')
			time.sleep(0.1)
			continue

		#print('expanding dims')
		frame_input = tf.expand_dims(frame, 0)
		#print('expanded dims')
		#print('resizing Img...')
		frame_input = resizeImage(frame_input, DEFAULT_SIZE,)
		#print('img resized')

		#print('predicting...')
		boxes, scores, classes, numbs = yolo.predict(frame_input)
		#print('predicted')
		
		frame = draw_outputs(frame,(boxes,scores,classes,numbs),class_name)

		cv2.imshow('yolo',frame)

		if cv2.waitKey(1) == ord('q') :
			cv2.destroyAllWindows()
			break


if __name__ == '__main__' :
	main()
