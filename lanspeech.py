from nltk import *
from nltk.corpus import *
from gtts import gTTS
from playsound import playsound
import os


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
    global lang
    ratios = lang_ratio(input)
    lang = max(ratios, key = ratios.get)
    return lang

ans = 'Y'


def speech(input):
    
    # text to speeech
    language = 'en'
    myobj = gTTS(text=input, lang=language, slow=False) 
    path = "C:/Users/Dell/Desktop/Be/Speech"
    complete_name = os.path.join(path, 'Speech.mp3')
    myobj.save(complete_name)
    #myobj.save("speech.mp3")
    #os.system("convert.wav")
   
    
    
    
    