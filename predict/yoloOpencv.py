import time
import cv2
import numpy as np

class opencvYOLO():
    def __init__(self, modeltype="yolov3", imgsize=(416,416), objnames="coco.names", weights="yolov3.weights", cfg="yolov3.cfg", score=0.25, nms=0.6):
        self.modeltype = modeltype
        self.imgsize = imgsize
        self.score = score
        self.nms = nms

        self.inpWidth = self.imgsize[0]
        self.inpHeight = self.imgsize[1]
        self.classes = None
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        dnn = cv2.dnn.readNetFromDarknet(cfg, weights)
        #dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        #dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.net = dnn

    def setScore(self, score=0.5):
        self.score = score

    def setNMS(self, nms=0.8):
        self.nms = nms

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs, labelWant, drawBox, bold, textsize, bcolor, tcolor):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
 
        classIds = []
        labelName = []
        confidences = []
        boxes = []
        boxbold = []
        labelsize = []
        boldcolor = []
        textcolor = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                label = self.classes[classId]
                if( (labelWant=="" or (label in labelWant)) and (confidence > self.score) ):
                    #print(detection)
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    if(left<0): left=0
                    top = int(center_y - height / 2)
                    if(top<0): top=0
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))
                    boxbold.append(bold)
                    labelName.append(label)
                    labelsize.append(textsize)
                    boldcolor.append(bcolor)
                    textcolor.append(tcolor)

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices

        nms_classIds = []
        #labelName = []
        nms_confidences = []
        nms_boxes = []
        nms_boxbold = []
        nms_labelNames = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            nms_confidences.append(confidences[i])
            nms_classIds.append(classIds[i])
            nms_boxes.append(boxes[i])
            nms_labelNames.append(labelName[i])

            if(drawBox==True):
                #print(boxbold[i], boldcolor[i], textcolor[i], labelsize[i])
                if(classIds[i]==1):
                    txt_color = (0,0,255,0)
                elif(classIds[i]==2):
                    txt_color = (255,255,0,0)
                else:
                    txt_color = (0,255,0,0)

                self.drawPred(frame, classIds[i], confidences[i], boxbold[i], txt_color, textcolor[i],
                    labelsize[i], left, top, left + width, top + height)

        self.bbox = nms_boxes
        self.classIds = nms_classIds
        self.scores = nms_confidences
        self.labelNames = nms_labelNames
        self.frame = frame

    # Draw the predicted bounding box
    def drawPred(self, frame, classId, conf, bold, boldcolor, textcolor, textsize, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), boldcolor, 2)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #top = max(top, labelSize[1])
        #cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, textsize, textcolor)

    def getObject(self, frame, labelWant=("car","person"), drawBox=False, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255)):
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        # Sets the input to the network
        net = self.net
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        self.postprocess(frame, outs, labelWant, drawBox, bold, textsize, bcolor, tcolor)
        self.objCounts = len(self.indices)
        # Put efficiency information. The function getPerfProfile returns the 
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    def listLabels(self):
        for i in self.indices:
            i = i[0]
            box = self.bbox[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            classes = self.classes
            #print("Label:{}, score:{}, left:{}, top:{}, right:{}, bottom:{}".format(classes[self.classIds[i]], self.scores[i], left, top, left + width, top + height) )

    def list_Label(self, id):
        box = self.bbox[id]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        classes = self.classes
        label = classes[self.classIds[id]]
        score = self.scores[id]

        return (left, top, width, height, label, score)
