# -*- coding: utf-8 -*-
"""
Script that 

@author: M
"""


# Dependencies: pip install requests beautifulsoup4 bs4 pandas selenium

import requests # for html requests
import time
import csv
from bs4 import BeautifulSoup # parses html data
#import json # parses data
#from pandas import DataFrame as df # organises data
from selenium import webdriver
import matplotlib as plot
import pandas as pd


# requests works fine to get robots.txt but for accessing other parts of the page we need selenium
robots= requests.get("https://www.systembolaget.se/robots.txt", verify=False) # get robots.txt to check for scrape permission
permissions = BeautifulSoup(robots.text, 'html.parser') # parse robots.txt
print(permissions) # print permissions and make sure request.get is not directed at disallowed path

# requests cannot bypass age restriction at systembolaget, use selenium instead to automate age verification
page = requests.get("https://www.systembolaget.se/sok/?categoryLevel1=Vin", verify=False) # verify = False is NOT recommended
                                                                                          # either change to True (default) or add path to certificate
                                                                                          
                                                                       
MAX_PAGES = 100
with open('data.csv', 'w') as d: #Write to csv file
    d.write('Wine, Price, Volume, Body, Tannins, Fruit, Country \n')
# webdriver requires Firefox drivers in current path, available at https://github.com/mozilla/geckodriver/releases
browser = webdriver.Firefox()  

browser.get('https://www.systembolaget.se/sok/?categoryLevel1=Vin&categoryLevel2=R%C3%B6tt%20vin')
time.sleep(3)  # allows for page to load before trying to locate and click buttons
# This block of code only has to be run once to gain access to the listings 
age_button = browser.find_element_by_class_name('css-1anj5eg') # Locate button for age restriction
age_button.click() # Click button for age restriction
time.sleep(1)
cookie_button = browser.find_element_by_class_name('css-1l6bdmi') # Locate button for accepting cookies
cookie_button.click() # Click button for accepting cookies
time.sleep(1)
shop_button = browser.find_element_by_class_name('css-18v1esa') # Locate button for updated webshop
shop_button.click() # Click button for updated webshop
time.sleep(1)

for page_num in range(MAX_PAGES):
    browser.get('https://www.systembolaget.se/sok/?categoryLevel1=Vin&categoryLevel2=R%C3%B6tt%20vin&page=' + str(page_num+1))
    time.sleep(1)  # allows for page to load before trying to locate and click buttons
    name        = browser.find_elements_by_class_name('css-uiubfo')
    price   = browser.find_elements_by_class_name('css-tz3s9q')
    cntry_volume= browser.find_elements_by_class_name('css-1ak0hsq')
    cntry        = cntry_volume[::2]
    vol_raw      = cntry_volume[1::2]
    # For the tasting profiles, the numerical values of "body", "tannins" and "fruit" are located in an aria-label associated with the graphics,
    # however, it is nested in a string and to convert it to a numerical value we first need to remove the known parts of the string.
    SK      = browser.find_elements_by_class_name('css-1q82610')
    body    = []
    tannins = []
    fruit   = []       
    for i in range(len(SK)):
        if 'Fyllighet' in SK[i].get_attribute('aria-label'):
            body.append(SK[i].get_attribute('aria-label').replace('Fyllighet har ett värde på ', '').replace(' av 12', '')) 
        if 'Strävhet'  in SK[i].get_attribute('aria-label'):
            tannins.append(SK[i].get_attribute('aria-label').replace('Strävhet har ett värde på ', '').replace(' av 12', ''))
        elif 'Sötma' in SK[i].get_attribute('aria-label'):       
            tannins.append('NA') 
        if 'Fruktsyra' in SK[i].get_attribute('aria-label'):
            fruit.append(SK[i].get_attribute('aria-label').replace('Fruktsyra har ett värde på ', '').replace(' av 12', ''))               
    vol     = []
    with open('data.csv', 'a') as d: #Write to csv file
        for i in range(len(name)):
            V  =  [int(s) for s in vol_raw[i].text.split()    if s.isdigit()]   # pick out numbers corresponding to volume (remove ML)
            if len(V) != 0:
                vol   = vol   + V            
            else:
                vol  = vol  + ['NA']
            if len(name[i].text) > 0:
                d.write(name[i].text + ',' + price[i].text.split(':')[0] + ',' + str(vol[i]) + ',' + body[i] + ',' + tannins[i] + ',' + fruit[i] + ',' + cntry[i].text + '\n')


#Close browser once data are collected
browser.close()


#Things to fix: 
        # Not all wines have tasting profiles, add conditional N/A if missing
        # Säljstart! (page 49) breaks country_year








