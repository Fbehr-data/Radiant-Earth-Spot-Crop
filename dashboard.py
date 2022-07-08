import numpy as np                  #unused
import pandas as pd                 #unused
import matplotlib.pyplot as plt     #unused
from fileinput import filename
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from PIL import Image, ImageTk
from io import BytesIO
import requests                     #unused
import sys                          #unused
import socket                       #unused



# Create Window and define size, title and background color

root = tk.Tk()
root.resizable(False, False)
root.geometry('1280x720')
root.title("Dashboard")
root.configure(bg='#c7d5e0')

# Create Top Bar and display the actual date and time

top_bg = tk.Canvas(root, width=1280, height=60, bg='#ff4a11', highlightthickness=0).place(x=0, y=0)
tk.Label(top_bg, text='Dude, were are my crops?', font='Montserrat 25', bg='#ff4a11', fg='white').place(x=15, y=3)
tk.Label(top_bg, text=datetime.now().strftime('%A, %d %B %Y'), font='Montserrat 18', bg='#ff4a11', fg='white').place(x=930, y=8)


# Create Box for Graphs

graph_box = tk.Canvas(root, width=300, height=600, bg='#2a475e', highlightthickness=0).place(x=10, y=90)
graph_box_top = tk.Canvas(root, width=300, height=35, bg='#1b2838', highlightthickness=0).place(x=10, y=70)
tk.Label(graph_box_top, text='Some Graphs', font='Montserrat 14 bold', bg='#1b2838', fg='#FFFFFF').place(x=15, y=70)

# Create Box for statisic

statistic_box = tk.Canvas(root, width=300, height=600, bg='#2a475e', highlightthickness=0).place(x=320, y=90)
statistic_box_top = tk.Canvas(root, width=300, height=35, bg='#1b2838', highlightthickness=0).place(x=320, y=70)
tk.Label(statistic_box_top, text='Some statistics', font='Montserrat 14 bold', bg='#1b2838', fg='#FFFFFF').place(x=330, y=70)


#fp = 'C:\Python\API\radient_logo.jpeg'
#filename = 'radient_logo.jpg'

#im = image(fp, filename)

#im = ImageTk.PhotoImage(Image.open("c:\Python\API\radient_logo.jpg"))
#canvas = tk.Canvas(width=461, height=193)
#canvas.place(x=1000, y=650) 
#Canvas_Image = canvas.create_image(im)

# Mainloop to close the window function

root.mainloop()
