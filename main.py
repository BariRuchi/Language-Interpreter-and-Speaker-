from tkinter import image_names
import streamlit as st
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

#import easyocr
from nltk import *
from nltk.corpus import *
from gtts import gTTS


#st.title("Language Interprter and Speaker")

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

model1 = tf.keras.models.load_model("Nmodel.h5")

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
    #st.write(letter, "hey")
    word = "".join(letter)
    return word




def load_image(image_file):
	img = Image.open(image_file)
	return img

def save_uploadedfile(uploadedfile):
     with open(os.path.join("C:/Users/Dell/Desktop/BE/Uploaded",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     #return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

#st.subheader("Image")
image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if image_file is not None:
    # To See details
	file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
	#st.write(image_file.name)

    # To View Uploaded Image
	st.image(load_image(image_file),width=250)

	save_uploadedfile(image_file)
    

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





path = "C:/Users/Dell/Desktop/Be/Uploaded"
valid_images = [".jpg",".gif",".png"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img= cv2.imread(os.path.join(path,f))
    #st.write("Yes it is there")

    #grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2_imshow(gray)
    cv2.waitKey(0)

    cv2.waitKey(0)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #cv2_imshow(thresh)
    cv2.waitKey(0)

    #dilation
    kernel = np.ones((5,50), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #cv2_imshow(img_dilation)
    cv2.waitKey(0)

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
        #st.write("Done")
        num+=1
    #st.write(num)

    #Removes extra spacing created in image
    if os.path.exists("C:/Users/Dell/Desktop/Be/output/1.png") is True:
        for i in range(1,num):
            image = cv2.imread("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
            #st.write("Yesssss")
            height = img.shape[0]
            width = img.shape[1]
            #width, height = image.size
        
            if (width< 75 and height<75):
                os.remove("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")

        #st.write("Yesssss")

    outputletters=[]
    num=3
    for i in range(1,num):
        if (os.path.exists("C:/Users/Dell/Desktop/Be/output/1.png") is True):
            #letter,image = get_letters("C:/Users/Dell/Desktop/Be/output/"+str(i)+".png")
            letter,image = get_letters("1.png")
            
            #st.write(letter) 
            #st.write(image)
            word = get_word(letter)
            outputletters.append(word)
            #print(word)
            #plt.imshow(image)
        
    #print(outputletters)
    #st.write(outputletters[1])
    sent = (' '.join(map(str,outputletters)))
    st.write(sent)

#Language Identification
def lang_ratio(input):
    lang_ratio={}
    tokens = wordpunct_tokenize(input)
    words = [word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        lang_ratio[language] = len(common_elements)
    return lang_ratio 

def detect_language(input):
    ratios = lang_ratio(input)
    lang = max(ratios, key = ratios.get)
    return lang
ans = 'Y'


lang = detect_language(sent)
#print("\n")
st.write(lang)
#print(sent+"\n Langauge: "+ lang)


# text to speeech
language = 'en'
    
myobj = gTTS(text=sent, lang=language, slow=False)
    
myobj.save("Demo.mp3")
    
outputletters.clear()

audio_dir ="Demo.mp3"

audio_file = open(audio_dir, "rb")

st.audio(audio_file.read())