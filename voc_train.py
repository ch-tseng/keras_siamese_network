# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from xml.dom import minidom
from sklearn.model_selection import train_test_split

class Config:
    DS_PATH = "/WORK1/MyProjects/for_Sale/Face_Mask_wear/dataset/masks_sunplusit_outsource_all/"
    # specify the shape of the inputs for our network
    IMG_SHAPE = (64, 64, 3)
    
    # specify the batch size and number of epochs
    BATCH_SIZE = 16
    EPOCHS = 300

    # define the path to the base output directory
    BASE_OUTPUT = "output"

    # use the base output path to derive the path to the serialized
    # model along with training history plot
    MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
    PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# instantiate the config class
config = Config()

def getLabels(xmlFile):
	labelXML = minidom.parse(xmlFile)
	labelName = []
	labelXmin = []
	labelYmin = []
	labelXmax = []
	labelYmax = []
	totalW = 0
	totalH = 0
	countLabels = 0

	tmpArrays = labelXML.getElementsByTagName("name")
	for elem in tmpArrays:
		labelName.append(str(elem.firstChild.data))

	tmpArrays = labelXML.getElementsByTagName("xmin")
	for elem in tmpArrays:
		labelXmin.append(int(elem.firstChild.data))

	tmpArrays = labelXML.getElementsByTagName("ymin")
	for elem in tmpArrays:
		labelYmin.append(int(elem.firstChild.data))

	tmpArrays = labelXML.getElementsByTagName("xmax")
	for elem in tmpArrays:
		labelXmax.append(int(elem.firstChild.data))

	tmpArrays = labelXML.getElementsByTagName("ymax")
	for elem in tmpArrays:
		labelYmax.append(int(elem.firstChild.data))

	return labelName, labelXmin, labelYmin, labelXmax, labelYmax


def make_pairs(images, labels):
	# print("TEST", images.shape, labels.shape)
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []

	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
	print('idx=', idx)

	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]

		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])

		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)

	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(128, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)

	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(256, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)

	# Added by CH, third set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(128, (2, 2), padding="same", activation="relu")(x)
	#x = MaxPooling2D(pool_size=2)(x)
	#x = Dropout(0.25)(x)


	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)

	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model

def load_ds(dir, ratio=0.2):
	img_folder = os.path.join(dir, "images")
	xml_folder = os.path.join(dir, "labels")
	folders, imgs, img_classes = [], [], []

	#print(img_folder)
	for file in os.listdir(img_folder):
		#print(file)
		filename, file_extension = os.path.splitext(file)
		file_extension = file_extension.lower()

		if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
			img_file = file
			xml_file = filename + ".xml"
			img_path = os.path.join(img_folder, img_file)
			xml_path = os.path.join(xml_folder, xml_file)

			print(xml_path)
			if(os.path.exists(xml_path)):
				image = cv2.imread(img_path)
				labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(xml_path)
				for id, class_name in enumerate(labelName):
					crop = image[labelYmin[id]:labelYmax[id], labelXmin[id]:labelXmax[id]]
					print("CROP:", crop.shape)
					imgs.append( cv2.resize(crop, (config.IMG_SHAPE[0], config.IMG_SHAPE[1])) )
					img_classes.append( class_name)


	class_map = {}
	class_list = list(set(img_classes))
	f = open("class_map.txt", 'w')
	for id, class_name in enumerate(class_list):
		class_map.update( {class_name:id} )
		f.write("{}:{}\n".format(class_name, id))
	f.close()

	classes = []
	for class_name in img_classes:
		classes.append(class_map[class_name])

	print("Class Map:", class_map)

	imgs = np.array(imgs)
	classes = np.array(classes)
	print(imgs.shape, classes.shape)

	#sys.exit(1)
	if(ratio>0):
		X_train, X_test, y_train, y_test = train_test_split(imgs, classes, test_size=ratio)
	else:
		X_train, X_test, y_train, y_test = imgs, classes, None, None
	#print("SHAPE1:", imgs.shape, classes.shape)
	#imgs = np.expand_dims(imgs, axis=0)
	#classes = np.expand_dims(classes, axis=0)
	#print("SHAPE2:", imgs.shape, classes.shape)

	return (X_train, y_train), (X_test, y_test)

(trainX, trainY), (testX, testY) = load_ds(config.DS_PATH)
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
#trainX = np.expand_dims(trainX, axis=-1)
#testX = np.expand_dims(testX, axis=-1)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)
print(trainX.shape, trainY.shape)

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featureExtractor.summary()

featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

model.summary()


# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE,
	epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)


