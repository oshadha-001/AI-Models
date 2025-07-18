import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import PIL
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

try:
    os.mkdir('data')
except:
    print('Path Already Exists')

width, height = 500, 500

win = tk.Tk()

font_btn = 'Helvetica 20 bold'
font_label = 'Helvetica 22 bold'
count = 0

import joblib

clsfr = joblib.load('sinhala-character-knn.sav')

label_dict = {0: 'අ', 1: 'එ', 2: 'ඉ', 3: 'උ'}


def event_function(event):
    x = event.x  # x coordinate of mouse pointer
    y = event.y  # y coordinate of mouse pointer

    x1 = x - 20
    y1 = y - 20

    x2 = x + 20
    y2 = y + 20

    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='black')


def save():
    global count

    img_array = np.array(img)
    img_array = cv2.resize(img_array, (8, 8))

    path = os.path.join('data', str(count) + '.jpg')
    # path=data/0.jpg

    cv2.imwrite(path, img_array)

    count = count + 1


def clear():
    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (width, height), (255, 255, 255))
    img_draw = ImageDraw.Draw(img)
    label_status.config(text='PREDICTED CHARACTER: NONE')


def predict():
    img_array = np.array(img)  # converting to numpy array
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)  # converting into a gray image
    img_array = cv2.resize(img_array, (8, 8))  # resizing into 8x8

    # plt.imshow(img_array,cmap='binary')

    img_array = np.reshape(img_array, (1, 64))  # reshaping into 1x64
    # img_array=img_array/255.0*15.0

    result = clsfr.predict(img_array)[0]
    label = label_dict[result]
    label_status.config(text='PREDICTED CHARACTER:' + label)

    plt.show()


canvas = tk.Canvas(win, width=width, height=height, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='green', fg='white', font=font_btn, command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font=font_btn, command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='orange', fg='white', font=font_btn, command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='red', fg='white', font=font_btn, command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='PREDICTED CHARACTER: NONE', bg='white', font=font_label)
label_status.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', event_function)
img = Image.new('RGB', (width, height), (255, 255, 255))
img_draw = ImageDraw.Draw(img)

win.mainloop()