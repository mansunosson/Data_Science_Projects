# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:09:34 2020

@author: M
"""

# Recommendation app for red wines


# Define function to recommend a wine
import pandas as pd
from pandasql import sqldf
from tkinter import *

# Create main window
root = Tk()
root.geometry("390x340")

def winequery(minprice, maxprice, body, tannins, fruit):
    try:
        wines = pd.read_csv(r'data.csv', encoding='ISO-8859-1')
        rootquery  = "SELECT * FROM wines WHERE "
        pricecond  = "ppf BETWEEN " + str(minprice) + " AND " + str(maxprice)
        bodycond   = "body=" + str(body)
        tanninscond= "tannins=" + str(tannins)
        fruitcond  = "fruit=" + str(fruit)
    
        q = rootquery + pricecond + " AND " + bodycond + " AND " + tanninscond + " AND " + fruitcond + ";"
    
        matches = sqldf(q,locals())
    
        if matches.shape[0] == 0:
            return "No such wine!"
        else: 
            randwine = matches.sample(1)
            return "You could try " + randwine.iloc[0]["wine"] + " from " + randwine.iloc[0]["country"] + " at " + str(int(round(randwine.iloc[0]["ppf"])))  + " per bottle."
    except:
        return "Make sure to use valid inputs"
    
def recommend():
    recwine = winequery(minpriceinput.get(), maxpriceinput.get(), bodyinput.get(), tanninsinput.get(), fruitinput.get())
    reclabel.configure(text = recwine)
    recbutton.configure(text = "Recommend another wine")

# Create label
versionlabel = Label(root, text = "RedCommend version 0.1")
versionlabel.pack()

reclabel = Label(root, text = "Hit the button to get a wine recommendation!")
reclabel.place(x = 25, y = 250)

# Create input boxes for price and flavour profiles
minpriceinput = Entry(root)
minpriceinput.place(x = 100, y = 50, width = 40, height = 20)
minpriceinput.insert(0, '0')

minpricelabel = Label(root, text = "Minimum price:")
minpricelabel.place(x = 5, y = 50)

maxpriceinput = Entry(root)
maxpriceinput.place(x = 100, y = 80, width = 40, height = 20)
maxpriceinput.insert(1, '999')

maxpricelabel = Label(root, text = "Maximum price:")
maxpricelabel.place(x = 5, y = 80)

bodyinput = Entry(root)
bodyinput.insert(1, '0-12')
bodyinput.place(x = 100, y = 120, width = 40, height = 20)

bodylabel = Label(root, text = "Body:")
bodylabel.place(x = 63, y = 120)

tanninsinput = Entry(root)
tanninsinput.place(x = 100, y = 150, width = 40, height = 20)
tanninsinput.insert(1, '0-12')

tanninslabel = Label(root, text = "Tannins:")
tanninslabel.place(x = 50, y = 150)

fruitinput = Entry(root)
fruitinput.place(x = 100, y = 180, width = 40, height = 20)
fruitinput.insert(1, '0-12')
fruitlabel = Label(root, text = "Fruit:")
fruitlabel.place(x = 66, y = 180)

# Create recommend button
recbutton = Button(root, text = "Recommend a wine", command = recommend)
recbutton.place(x = 25, y = 220)

# 
root.mainloop()