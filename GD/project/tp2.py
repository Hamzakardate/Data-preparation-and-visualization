# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:25:18 2021

@author: ASUS


from tkinter import *

fen= Tk()

l=Label(fen,text="hello world")
l.pack()

b=Button(fen,text="afficher")
b.pack()


fen.mainloop()
"""
from tkinter import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys    
from tkinter import filedialog



def importer():
    
    
    
    filename = filedialog.askopenfilename()
    global df

    df = pd.read_excel(filename)

       
        
           
    


def afficher_maunquant():
    
    root2 = Tk()
    T = Text(root2, height = 50, width = 50)
    T.pack()
    f=df.isnull().sum()
    T.insert(END, f)
    root.mainloop()
    

def afficher():
    root1 = Tk()
    T = Text(root1, height = 50, width = 170)
    T.pack()
    a=""
    f=df.head().to_numpy()
    for i in range(len(f)):
        for j in range(len(f[0])):
            a+=str(f[i][j])
            a+="\t"
        a+="\n"
    T.insert(END, a)
    root1.mainloop()

def pm():
    root3 = Tk()
    T = Text(root3, height = 20, width = 20)
    T.pack()
    missing_values_count = df.isnull().sum()
    missing_values_count
    a=""
    # combient des VM
    total_cells = np.product(df.shape)
    a+=str(total_cells)
    a+="\n"
    total_missing = missing_values_count.sum()
    a+=str(total_missing)
    a+="\n"
    # percent vm
    percent_missing = (total_missing/total_cells) * 100
    a+=str(percent_missing)
    a+="%"
    T.insert(END, a)
    root3.mainloop()

def corr():
    root4 = Tk()
    T = Text(root4, height = 50, width = 170)
    T.pack()
    f=df.corr()
    T.insert(END, f)
    root4.mainloop()

def pvar():
    root5 = Tk()
    T = Text(root5, height = 50, width = 170)
    T.pack()
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    T.insert(END, missing_value_df)
    root4.mainloop()
    
root = Tk()
root.geometry("500x500")
btn_height = Button(root, text="Importer",command=importer)
btn_height.place(height=50,width=60, x=200, y=50)

btn_width = Button(root, text="afficher",command=afficher)
btn_width.place(height=50,width=60, x=50, y=50)
btn_relheight = Button(root, text="afficher les valeurs maunquant",command=afficher_maunquant)
btn_relheight.place(height=50,width=180, x=300, y=50)
btn_relwidth= Button(root, text="Pourcentage des valeurs manquantes" ,command=pm)
btn_relwidth.place(height=50,width=240, x=5, y=150)
btn_relx=Button(root, text="La corrélation",command=corr)
btn_relx.place(height=50,width=100, x=180, y=250)
btn_rely=Button(root, text="Pourcentage de chaque valeurs manquantes",command=pvar)
btn_rely.place(height=50,width=240, x=255, y=150)

root.mainloop()
