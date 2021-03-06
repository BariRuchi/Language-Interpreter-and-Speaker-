# -*- coding: utf-8 -*-
"""AllInOne.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1seWlHn7g5DONFMIvtED-HIA3A38431D3
"""

! pip install -q kaggle

from google.colab import files

files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d vaibhao/handwritten-characters

! mkdir train1
! unzip handwritten-characters.zip -d train1

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import random 
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import AveragePooling2D
from tensorflow.keras.optimizers import SGD
import IPython

import tensorflow as tf
import cv2
import numpy
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
import imutils
import numpy as np
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import os
from PIL import Image
import matplotlib.pyplot as plt
import IPython

!pip install easyocr
import nltk
nltk.download('stopwords')

!pip install gTTS
!pip uninstall opencv-python-headless==4.5.5.62
!pip install opencv-python-headless==4.5.2.52

import easyocr
from nltk import *
from nltk.corpus import *
from gtts import gTTS

#Training And Validation Dataset
dir = "/content/train1/Train"
train_data = []
img_size = 32
non_chars = ["#","$","&","@"]
for i in os.listdir(dir):
    if i in non_chars:
        continue
    count = 0
    sub_directory = os.path.join(dir,i)
    for j in os.listdir(sub_directory):
        count+=1
        if count > 4000:
            break
        img = cv2.imread(os.path.join(sub_directory,j),0)
        img = cv2.resize(img,(img_size,img_size))
        train_data.append([img,i])



val_dir = "/content/train1/Validation"
val_data = []
img_size = 32
for i in os.listdir(val_dir):
    if i in non_chars:
        continue
    count = 0
    sub_directory = os.path.join(val_dir,i)
    for j in os.listdir(sub_directory):
        count+=1
        if count > 1000:
            break
        img = cv2.imread(os.path.join(sub_directory,j),0)
        img = cv2.resize(img,(img_size,img_size))
        val_data.append([img,i])

random.shuffle(train_data)
random.shuffle(val_data)

train_X = []
train_Y = []
for features,label in train_data:
    train_X.append(features)
    train_Y.append(label)


val_X = []
val_Y = []
for features,label in val_data:
    val_X.append(features)
    val_Y.append(label)

import pickle

import pickle

pickle_out = open("train_x.pickle","wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_in = open("train_x.pickle","rb")
train_x = pickle.load(pickle_in)

pickle_outy = open("train_y.pickle","wb")
pickle.dump(train_Y, pickle_outy)
pickle_outy.close()

pickle_iny = open("train_y.pickle","rb")
train_y = pickle.load(pickle_iny)

pickle_out1 = open("val_x.pickle","wb")
pickle.dump(val_X, pickle_out1)
pickle_out1.close()

pickle_in1 = open("val_x.pickle","rb")
val_x = pickle.load(pickle_in1)

pickle_out1y = open("val_y.pickle","wb")
pickle.dump(val_Y, pickle_out1y)
pickle_out1y.close()

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
print(train_X.shape,val_X.shape)

#Model creation
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape = (32,32,1), dtype = 'float32')
x = Conv2D(28, kernel_size = (5,5), padding = 'same', activation = 'relu')(inputs)
x = Conv2D(28, kernel_size = (5,5), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Dropout(0.25)(x)
x = Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
x = Conv2D(32, kernel_size = (3,3), padding = 'same', activation = 'relu')(x)
x = Conv2D(54, kernel_size = (5,5), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(256)(x)
x = Dropout(0.25)(x)
output = Dense(35, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = output)

model.summary()

#AlexNet

model = Sequential()

model.add(Conv2D(96, (11, 11), padding = "same", activation='relu', input_shape=(32,32,1)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(256, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Conv2D(384, (2, 2), activation='relu'))
#model.add(Conv2D(384, (1, 1), activation='relu'))
model.add(Conv2D(256, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(35, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
history = model.fit(train_X,train_Y, epochs=25, batch_size=150, validation_data = (val_X, val_Y),  verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



model = tf.keras.models.load_model("/content/model Al-2.h5")

#Contours and getting letters forhandwritten image text

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
          ypred = model.predict(thresh)
          #print(ypred[0])
          ypred = LB.inverse_transform(ypred)
          #print(ypred[0])
          [x] = ypred
          letters.append(x)
    return letters, image

#plt.imshow(image)
def get_word(letter):
    word = "".join(letter)
    return word



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

def digital_ip(path):
  #digital image to text
  readme = easyocr.Reader(['en'])
  
  texts = readme.readtext(path)
  
  result=''
  
  for text in texts:
    result += text[1]
  
  input1 = result
  lang = detect_language(input1)
  print(input1+"\n Langauge: "+ lang)
  #print(result)
  

  # text to speeech
  language = 'en'

  myobj = gTTS(text=result, lang=language, slow=False)
 
  myobj.save("speech.mp3")
  


def hand_ip(path):
  print("\n")

  #Segmentation and Image processing

  #import image
  image = cv2.imread(path)


  #grayscale
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  cv2_imshow(gray)
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

  im = cv2.resize(image,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
  num=1

  if (os.path.isdir("/content/output") is True ):
    pass
  else:
    !mkdir output

  for i, ctr in enumerate(sorted_ctrs):
      # Get bounding box 
      x, y, w, h = cv2.boundingRect(ctr)

      # Getting ROI
      roi = image[y:y+h, x:x+w]

      # show ROI
      #cv2_imshow(roi)
      
      cv2.imwrite('/content/output/%d.png'%(num), roi)
      num+=1



  #Removes extra spacing created in image

  if os.path.exists("/content/output/1.png" ) is True:
    for i in range(1,num):
      image = Image.open("/content/output/"+str(i)+".png")
      width, height = image.size

      if (width< 75 and height<75):
        os.remove("/content/output/"+str(i)+".png")
    #print("Done")


  outputletters=[]
  for i in range(1,num):
    if (os.path.exists("/content/output/"+str(i)+".png") is True):
      letter,image = get_letters("/content/output/"+str(i)+ ".png")
      word = get_word(letter)
      outputletters.append(word)
      #print(word)
      #plt.imshow(image)
    
  #print(outputletters)

  sent = (' '.join(map(str, outputletters)))
  #print(sent)

  lang = detect_language(sent)
  print("\n")
  print(sent+"\n Langauge: "+ lang)
  #print(result)
  
  # text to speeech
  language = 'en'
  
  myobj = gTTS(text=sent, lang=language, slow=False)
  
  myobj.save("Demo.mp3")
  
  outputletters.clear()

LB = LabelBinarizer()
train_Y = LB.fit_transform(train_Y)
val_Y = LB.fit_transform(val_Y)
train_X = np.array(train_X)/255.0
train_X = train_X.reshape(-1,32,32,1)
train_Y = np.array(train_Y)
val_X = np.array(val_X)/255.0
val_X = val_X.reshape(-1,32,32,1)
val_Y = np.array(val_Y)

print("-----LANGUAGE INTERPRETER AND SPEAKER-----\n")
print("Select an appropriate option as shown below:")
print("D:For digital image text\nH:For handwritten image text")
print('\n')
num = input ("Enter which image text you want to try :") 
if num == 'D':
  upath = input("Enter image path of the picture:")
  print("\n")
  image = cv2.imread(upath)
  #imS = cv2.resize(image, (250, 150)) 
  cv2_imshow(image)
  print("\n")
  digital_ip(upath)
elif num == 'H':
  upath = input("Enter image path of the picture:")
  #print(upath)
  hand_ip(upath)
else:
  print("Enter valid option")
  error = 1 
  
#if (num == 'D' or num == 'H'):
  #print("hiii")
  #Text to speech

if error > 0:
  if os.path.exists("/content/speech.mp3"):
    a = "speech.mp3"
  else:
    a = "Demo.mp3"
    #print(a)
  path = '/content/'+a
  print("\n")
  IPython.display.Audio(path)

os.remove(path)


if os.path.exists("/content/output") is True:
  #os.remove("/content/output")
  for i in range(1,num):
    if (os.path.exists("/content/output/"+str(i)+".png") is True):
      os.remove("/content/output/"+str(i)+".png")     
      

print("Done")



