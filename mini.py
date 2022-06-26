from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from playsound import playsound
from PIL import Image, ImageTk
import os
from connect import hand_ip
from connect import *
from digital import digital_ip
import connect as c1
import digital as dg
import lanspeech as ls
import pygame
import pyttsx3

ws = Tk()
ws.title('LI&S')
ws.geometry('400x300')


def viewSelected():
    choice  = var.get()
    global output
    if choice == 1:
       output = "H"

    elif choice == 2:
       output = "D"
        
    return output

b1 = Button(ws, text='Upload', width=15,command = lambda:upload_file(),pady=5).pack() 

lbl = Label(text = "Select Option:").pack(pady= 10)
var = IntVar()
Radiobutton(ws, text="Handwritten", variable=var, value=1, command=viewSelected).pack()
Radiobutton(ws, text="Digital", variable=var, value=2, command=viewSelected).pack()
  
b2 = Button(ws, text='Process', width=15,command = lambda:process(),pady=5).pack()

def openNewWindow():
     
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(ws)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("New Window")
 
    # sets the geometry of toplevel
    newWindow.geometry("800x500")
    
    img=Image.open(complete_name)
    img=ImageTk.PhotoImage(img)
    e1 =tk.Label(newWindow)
    e1.pack(pady= 20)
    e1.image = img
    e1['image']=img # garbage collection
    
    global l, bp, ll
    if (output == 'H') :
        hand_ip()
        #print(c1.sent)
 
        # Create label
        l = Label(newWindow , text = c1.sent)
        l.config(font =("Courier", 14))
        l.pack(pady= 10)
        
        ll = Label(newWindow , text = ls.lang)
        ll.config(font =("Courier", 14))
        ll.pack(pady= 10)
        

        # Define a function to play the music
        def play_sound():
        
           engine = pyttsx3.init()
           rate = engine.getProperty('rate') 
           #print(rate)
           engine.setProperty('rate', 125)
           volume = engine.getProperty('volume')
           #print(volume)
           engine.setProperty('volume',1.0)  
           voices = engine.getProperty('voices')
           engine.setProperty('voice', voices[0].id)
           engine.say(c1.sent)
           engine.runAndWait()

        def Close():
            newWindow.destroy()

        # Add a Button widget        
        bp = Button(newWindow , text="Play",  width=10, bg="green",fg="white" ,command=play_sound).pack()
        Button(newWindow ,text="Exit", width=10,fg="black" ,command=Close).pack()
    
    if (output == 'D'):
        digital_ip()
        #print(dg.result)
        
        l = Label(newWindow , text = dg.result)
        l.config(font =("Courier", 14))
        l.pack(pady= 10)
        
        ll = Label(newWindow , text = ls.lang)
        ll.config(font =("Courier", 14))
        ll.pack(pady= 10)
        

        # Define a function to play the music
        def play_sound():
           global engine
           engine = pyttsx3.init()
           rate = engine.getProperty('rate') 
           #print(rate)
           engine.setProperty('rate', 125)
           volume = engine.getProperty('volume')
           #print(volume)
           engine.setProperty('volume',1.0)  
           voices = engine.getProperty('voices')
           engine.setProperty('voice', voices[0].id)
           engine.say(dg.result)
           engine.runAndWait()
         
        def Close():
            newWindow.destroy()    

        # Add a Button widget        
        bp = Button(newWindow , text="Play",  width=10, bg="green",fg="white" ,command=play_sound).pack()
        Button(newWindow ,text="Exit", width=10,fg="black" ,command=Close).pack()
          
    

def upload_file():
        global f,img_name,img,complete_name
        f_types = [('Jpg Files', '*.jpg'),
        ('PNG Files','*.png'),
        ('Jpeg Files', '*.jpeg')]   # type of files to select 
        filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
        #print(filename)
        for f in filename:
            #print(f)
            x = f.split("/")
            #print(x)
            img_name = x[-1]
            #print(img_name)
            img=Image.open(f) # read the image file
            path = "C:/Users/Dell/Desktop/BE/Uploaded"
            complete_name = os.path.join(path, img_name)
            img.save(complete_name)
          
        
def process():
    openNewWindow()

     
def clean():

    if os.path.exists("C:/Users/Dell/Desktop/Be/Speech/Speech.mp3"):
        os.remove("C:/Users/Dell/Desktop/Be/Speech/Speech.mp3")
    else:
        pass
   
   
    path = "C:/Users/Dell/Desktop/Be/Uploaded"
    print(img_name)
    complete_name = os.path.join(path,img_name)
    if os.path.exists(complete_name):
        os.remove(complete_name)
    else:
        pass
        
        
    if output =='H':
        
   
        c1.num = c1.num-1
   
        while (c1.num != 0):
            if os.path.exists("C:/Users/Dell/Desktop/Be/output/"+str(c1.num)+".png"):
                os.remove("C:/Users/Dell/Desktop/Be/output/"+str(c1.num)+".png")
                c1.num = c1.num - 1
            else:
                c1.num = c1.num - 1
               

  

b3 = Button(ws, text='Clear', width=15,command =lambda:clean(),pady=5).pack()


ws.mainloop()