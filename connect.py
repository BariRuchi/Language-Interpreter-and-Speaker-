#from tkinter import image_names
import os
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
import numpy
import imutils
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from numpy import asarray
import os
from PIL import Image
import seaborn as sns
import pickle
from lanspeech import *
import pygame


pickle_in = open("train_x.pickle","rb")
train_x = pickle.load(pickle_in)

pickle_iny = open("train_y.pickle","rb")
train_y = pickle.load(pickle_iny)

pickle_in1 = open("val_x.pickle","rb")
val_x = pickle.load(pickle_in1)

pickle_in1y = open("val_y.pickle","rb")
val_y = pickle.load(pickle_in1y)


LB = LabelBinarizer()
train_Y = LB.fit_transform(train_y)
val_Y = LB.fit_transform(val_y)
train_X = np.array(train_x)/255.0
train_X = train_X.reshape(-1,32,32,1)
train_Y = np.array(train_y)
val_X = np.array(val_x)/255.0
val_X = val_X.reshape(-1,32,32,1)
val_Y = np.array(val_Y)

model1 = tf.keras.models.load_model("model Al-2.h5")

def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
        if(type(x) == numpy.str_):
         pass
        else:
          roi = gray[y:y+h, x:x+w]
          thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
          thresh = cv2.resize(thresh, (32,32), interpolation = cv2.INTER_CUBIC)
          thresh = thresh.astype("float32") / 255.0
          thresh = np.expand_dims(thresh, axis=-1)
          thresh = thresh.reshape(1,32,32,1)
          ypred = model1.predict(thresh)
          #st.write(ypred)
          ypred = LB.inverse_transform(ypred)
          [x] = ypred
          letters.append(x)
    return letters, image

#plt.imshow(image)
def get_word(letter):
    #cv2.imshow(letter)
    word = "".join(letter)
    return word


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def hand_ip():
    global sent, num
    path = "C:/Users/Dell/Desktop/Be/Uploaded"
    valid_images = [".jpg",".png",".jpeg"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img= cv2.imread(os.path.join(path,f))

        #grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2_imshow(gray)
        #cv2.waitKey(0)

        #cv2.waitKey(0)

        #binary
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        #cv2_imshow(thresh)
        #cv2.waitKey(0)

        #dilation
        kernel = np.ones((5,50), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #cv2_imshow(img_dilation)
        #cv2.waitKey(0)

        #find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        im = cv2.resize(img,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
        num=1


        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box 
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = img[y:y+h, x:x+w]

            # show ROI
            #cv2_imshow(roi)
                
            cv2.imwrite("C:/Users/Dell/Desktop/Be/output/%d.png"%(num), roi)
            num+=1

        #Removes extra spacing created in image
        if os.path.exists("C:/Users/Dell/Desktop/Be/output/1.png") is True:
            for i in range(1,num):
                '''image = cv2.imread("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
                #st.write("Yesssss")
                #height = img.shape[0]
                #width = img.shape[1]
                width, height = int(image.size)
                '''
                img = pygame.image.load("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
                width = img.get_width()
                height = img.get_height()
                print(width)
                print(height)
            
                if (width< 75 and height<75):
                    os.remove("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
                    print("Yess done")

            #st.write("Yesssss")

        outputletters=[]
        for i in range(1,num):
            if (os.path.exists("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png") is True):
                letter,image = get_letters("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
                #letter,image = get_letters("1.png")
                
                word = get_word(letter)
                outputletters.append(word)
                print(word)
                #plt.imshow(image)
            
        print(outputletters)
        sent = (' '.join(map(str,outputletters)))
        #print(sent)
        
        detect_language(sent)
        
        speech(sent)
        
        return sent
        
