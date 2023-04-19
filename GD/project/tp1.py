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
from pandas import Series, DataFrame
import matplotlib as mpl
import seaborn as sns 
from tkinter.filedialog import *
import webbrowser
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
import string
import re
from nltk.corpus import stopwords
from itertools import chain
import warnings
import statsmodels as sm
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler
from mlxtend.preprocessing import minmax_scaling
warnings.filterwarnings('ignore')
# set seed for reproducibility
np.random.seed(0)

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


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
    root5.mainloop()
    
def create_plot():
    sns.set(style="white")
    figure,ax=plt.subplots()
    sns.countplot(df['Interested in Credit Card'],ax=ax)
    ax.set_title("Original Data")
    return figure

def preparation_donnees():
    root6 =Tk()
    root.wm_title("Embedding in Tk")
    
    label =Label(root6, text="Matplotlib with Seaborn in Tkinter")
    label.pack()
    
    fig = create_plot()
    
    canvas = FigureCanvasTkAgg(fig, master=root6)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    button =Button(root6, text="Quit", command=root6.destroy)
    button.pack()

    root6.mainloop()
    
def Equilibrer_donnees_plot():
    sns.set(style="white")
    figure,ax=plt.subplots()
    #preparation des donnees 
    y = df.iloc[:,-1]
    #X (les autres) sont les variables qui précèdent la dernière
    X= df.iloc[:,:-1]
    # Division de la bd
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    # Sur-échantillonnage
    rOs = RandomOverSampler()
    X_ro, y_ro = rOs.fit_resample(x_train, y_train)
    x_rtest, y_rtest = rOs.fit_resample(x_test, y_test)
    sb.countplot(x=y_ro,data=X_ro)
    ax.set_title("Original Data")
    return figure
    
def Equilibrer_donnees():
    root7 =Tk()
    root.wm_title("Embedding in Tk")
    
    label =Label(root7, text="Matplotlib with Seaborn in Tkinter")
    label.pack()
    
    fig = Equilibrer_donnees_plot()
    
    canvas = FigureCanvasTkAgg(fig, master=root7)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    button =Button(root7, text="Quit", command=root7.destroy)
    button.pack()

    root7.mainloop()
    
def fun_fig1():
    sns.set(style="white")
    
    

    original_data = pd.DataFrame(df["RFM Score"])
    # scale the goals from 0 to 1
    scaled_data = minmax_scaling(original_data, columns=["RFM Score"])
    # plot the original & scaled data together to compare
    
    fig, ax=plt.subplots(1,2,figsize=(15,3))
    sb.distplot(df["RFM Score"], ax=ax[0])
    ax[0].set_title("Original Data")
    sb.distplot(scaled_data, ax=ax[1])
    ax[1].set_title("Scaled data")
    return fig
    
 
def fun_fig2():
    sns.set(style="white")
    index_of_positive_pledges = df["RFM Score"] > 0
    
    # get only positive pledges (using their indexes)
    positive_pledges = df["RFM Score"].loc[index_of_positive_pledges]
    
    # normalize the pledges (w/ Box-Cox)
    normalized_pledges = pd.Series(stats.boxcox(positive_pledges)[0], 
                                   name='RFM Score', index=positive_pledges.index)
    
    # plot both together to compare
    fig, ax=plt.subplots(1,2,figsize=(15,3))
    sb.distplot(positive_pledges, ax=ax[0])
    ax[0].set_title("Original Data")
    sb.distplot(normalized_pledges, ax=ax[1])
    ax[1].set_title("Normalized data")
    return fig

def Normalize_donnees():
    root8 =Tk()
    root8.geometry("1000x1000")
    root8.wm_title("Embedding in Tk")
    
    label1 =Label(root8, text="Matplotlib with Seaborn in Tkinter")
    label1.pack()
    
    fig1 = fun_fig1()
    
    canvas = FigureCanvasTkAgg(fig1, master=root8)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    label2 =Label(root8, text="Matplotlib with Seaborn in Tkinter")
    label2.pack()
    
    fig2 = fun_fig2()
    
    canvas = FigureCanvasTkAgg(fig2, master=root8)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    button =Button(root8, text="Quit", command=root8.destroy)
    button.pack()

    root8.mainloop()
    
def remplacer_valeurs_manquantes():
    percent_missing = df.isnull().sum() * 100 / len(df)
    for i in range(len(percent_missing)):
        if percent_missing[i] <5 and percent_missing[i] !=0:
            df[df.columns[i]].fillna(df[df.columns[i]].mode()[0] ,inplace=True)
        else:
            # drop rows with missing values
            df.dropna(inplace=True)
    
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
btn_relx=Button(root, text="Etat des données ",command=preparation_donnees )
btn_relx.place(height=50,width=150, x=20, y=250)
btn_relx=Button(root, text="Equilibrer les données ",command=Equilibrer_donnees )
btn_relx.place(height=50,width=150, x=300, y=250)
btn_relx=Button(root, text="Normaliser les données ",command=Normalize_donnees )
btn_relx.place(height=50,width=150, x=20, y=350)
btn_relx=Button(root, text="Remplacer ou supprimer les valeurs manquantes ",command=remplacer_valeurs_manquantes )
btn_relx.place(height=50,width=350, x=180, y=350)
button =Button(root, text="Quitter", command=root.destroy)
button.pack()
root.mainloop()
