from turtle import numinput
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import bleedfacedetector as fd

def init_emotion(model):
    global net, emotions
    emotions = ['Confused', 'Relaxed', 'Stressed', 'Disgust', 'Fear']

eml = []

def emotion(image, returndata=False):
    img_copy = image.copy()
    faces = fd.ssd_detect(img_copy, conf=0.2)
    padding = 3
    test = []
    for x, y, w, h in faces:
        face = img_copy[y-padding:y+h+padding, x-padding:x+w+padding]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray, (64, 64))
        processed_face = resized_face.reshape(1, 1, 64, 64)
        net.setInput(processed_face)
        Output = net.forward()
        expanded = np.exp(Output - np.max(Output))
        probablities = expanded / expanded.sum()
        prob = np.squeeze(probablities)
        predicted_emotion = emotions[prob.argmax()]
        #print(predicted_emotion)
        test.append(predicted_emotion)
    eml.append(test[0])

    if returndata:
        return
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_copy[:, :, ::-1])
        plt.axis("off")


model = 'emotion-ferplus-8.onnx'
net = cv2.dnn.readNetFromONNX(model)

emotions = ['Confused', 'Relaxed', 'Stressed', 'Disgust', 'Fear']

i=0
def run():
    for i in range(1, 7):
        image = cv2.imread('media/img{}.jpeg'.format(i))
        emotion(image)
    print(np.array(eml))


#run()
#print(eml)
#print(np.array(eml))
