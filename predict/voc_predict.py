# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from xml.dom import minidom
from yoloOpencv import opencvYOLO
import os

MODEL_PATH = "../output/siamese_model/"
media = "/WORK1/MyProjects/for_Sale/Face_Mask_wear/demo/videos/1.mp4"
predict_size = (64, 64)
write_output = True
output_video_path = "output.avi"
video_rate = 24.0

yolo_face_detect = opencvYOLO(modeltype="yolov3", \
	objnames="models/face_detect/yolov3/obj.names", \
	weights="models/face_detect/yolov3/weights/yolov3_268000.weights",\
	cfg="models/face_detect/yolov3/yolov3.cfg",\
	score=0.25, nms=0.6)

#---------------------------------------------------------------------------------------

camera = cv2.VideoCapture(media)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("This video's resolution is: %d x %d" % (width, height))
if(write_output is True):
	#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))

def getFaces(img):
	yolo_face_detect.getObject(img, labelWant="", drawBox=False, bold=1, textsize=0.6, bcolor=(0,255,0), tcolor=(0,0,255))
	#print ("Object counts:", yolo_face_detect.objCounts)
	#yolo.listLabels()
	#print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
	#	format(len(yolo_face_detect.classIds), len(yolo_face_detect.scores), len(yolo_face_detect.labelNames), len(yolo_face_detect.bbox)) )

	return yolo_face_detect.labelNames, yolo_face_detect.bbox, yolo_face_detect.scores


def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def preprocess_img(img):
	img = cv2.resize(img, predict_size)
	img = np.expand_dims(img, axis=0)
	img = img / 255.0

	return img

def get_avg_score(class_folder, face_img):
	np_face1 = preprocess_img(face_img)

	count = 0
	total_scores = 0.0
	for file in os.listdir(class_folder):
		filename, file_extension = os.path.splitext(file)
		file_extension = file_extension.lower()

		if(file_extension.lower() in [".jpg",".jpeg",".png"]):
			np_face2 = preprocess_img(cv2.imread(os.path.join(class_folder, file)))
			preds = model.predict([np_face1, np_face2])
			proba = preds[0][0]
			total_scores += proba
			count += 1
			#print(proba)

	avg = total_scores/count
	#print(avg)
	return avg

# load the model from disk
print("[INFO] loading siamese model...")
model = load_model(MODEL_PATH)

model.summary()

#grabbed = True
i = 0
(grabbed, frame) = camera.read()
while grabbed:
	p_labels, p_bboxes, p_scores = getFaces(frame)

	for id, box in enumerate(p_bboxes):
		(x,y,w,h) = box
		face_area = frame[y:y+h, x:x+w]
		print("BOX:", box)
		if(not w*h>900):
			continue

		avg_good = get_avg_score('target_compare3/good/', face_area)
		avg_bad = get_avg_score('target_compare3/bad/', face_area)
		avg_none = get_avg_score('target_compare3/none/', face_area)
		score_mask = [avg_good, avg_bad, avg_none]
		class_id = score_mask.index(max(score_mask))
		if(class_id==0):
			class_name = "correct"
			txt_color = (0,255,0)
		elif(class_id==1):
			class_name = "incorrect"
			txt_color = (255,0,0)
		else:
			class_name = "no mask"
			txt_color = (0,0,255)

		frame = cv2.rectangle(frame, (x, y), (x+w, y+h), txt_color, 2)
		cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 2)

		if(write_output is True):
			out.write(frame)

		print(avg_good, avg_bad, avg_none)

	i += 1
	(grabbed, frame) = camera.read()

out.release()
camera.release()

