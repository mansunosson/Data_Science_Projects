# -*- coding: utf-8 -*-
"""
Script that collects information on wines from systembolaget

@author: M
"""


# Dependencies: pip install requests beautifulsoup4 bs4 pandas selenium


import time
from selenium import webdriver



# # requests works fine to get robots.txt but for accessing other parts of the page we need selenium
# # UNCOMMENT this block of code to check robots.txt for permissions
#import requests # for html requests
#from bs4 import BeautifulSoup # parses html data
#robots= requests.get("https://www.systembolaget.se/robots.txt", verify=False) # get robots.txt to check for scrape permission
#permissions = BeautifulSoup(robots.text, 'html.parser') # parse robots.txt
#print(permissions) # print permissions and make sure request.get is not directed at disallowed path

# # requests cannot bypass age restriction at systembolaget, use selenium instead to automate age verification
#page = requests.get("https://www.systembolaget.se/sok/?categoryLevel1=Vin", verify=False) # verify = False is NOT recommended   
                                                                       
MAX_PAGES = 70 #choose maximum number of pages to scrape
with open('data.csv', 'w') as d: # initialise headers of output csv file
    d.write('Wine, Price, Volume, Body, Tannins, Fruit, Country \n')
# webdriver requires Firefox drivers in current path, available at https://github.com/mozilla/geckodriver/releases
browser = webdriver.Firefox()  # initialise web browser

browser.get('https://www.systembolaget.se/') # open page in browser to verify age and accept cookies 
time.sleep(1)  # allows for page to load before trying to locate and click buttons
# This block of code only has to be run once to gain access to the listings 
age_button = browser.find_element_by_class_name('css-1anj5eg') # Locate button for age restriction
age_button.click() # Click button for age restriction
time.sleep(0.5)
cookie_button = browser.find_element_by_class_name('css-1l6bdmi') # Locate button for accepting cookies
cookie_button.click() # Click button for accepting cookies
time.sleep(0.5)
shop_button = browser.find_element_by_class_name('css-18v1esa') # Locate button for updated webshop
shop_button.click() # Click button for updated webshop
time.sleep(0.5)



for page_num in range(MAX_PAGES):
    browser.get('https://www.systembolaget.se/sok/?categoryLevel1=Vin&categoryLevel2=R%C3%B6tt%20vin&page=' + str(page_num+1)) # go to page_num+1
    time.sleep(0.5)  # allows for page to load before trying to locate and click buttons
    name        = browser.find_elements_by_class_name('css-uiubfo') # store names of wines
    price   = browser.find_elements_by_class_name('css-tz3s9q')     # store prices
    cntry_volume= browser.find_elements_by_class_name('css-1ak0hsq')# store country and volume
    cntry        = cntry_volume[::2]                                # extract country from box
    vol_raw      = cntry_volume[1::2]                               # extract volume from box
    # For the tasting profiles, the numerical values of "body", "tannins" and "fruit" are located in an aria-label associated with the graphics,
    # however, it is nested in a string and to convert it to a numerical value we first need to remove the known parts of the string.
    SK      = browser.find_elements_by_class_name('css-1q82610')    # store tasting profiles
    if len(SK) == len(name)*3: #if not all wines on page have tasting profile then go to next page
        body    = []    # initialise categories for tasting profiles
        tannins = []
        fruit   = []       
        for i in range(len(SK)): # loop over all wines on the page and extracts the numerical values for each category from the corresponding aria-label
            if 'Fyllighet' in SK[i].get_attribute('aria-label'):
                body.append(SK[i].get_attribute('aria-label').replace('Fyllighet har ett värde på ', '').replace(' av 12', ''))
            if 'Strävhet'  in SK[i].get_attribute('aria-label'):
                tannins.append(SK[i].get_attribute('aria-label').replace('Strävhet har ett värde på ', '').replace(' av 12', ''))
            elif 'Sötma' in SK[i].get_attribute('aria-label'): # sweet red wines are graded on sweetness instead of tannins, resulting in tannins = NA   
                tannins.append('NA') 
            if 'Fruktsyra' in SK[i].get_attribute('aria-label'):
                fruit.append(SK[i].get_attribute('aria-label').replace('Fruktsyra har ett värde på ', '').replace(' av 12', ''))               
        vol     = []
        with open('data.csv', 'a') as d: # prepare csv file for appending
            for i in range(len(name)):
                V  =  [int(s) for s in vol_raw[i].text.split()    if s.isdigit()]   # pick out numbers corresponding to volume (remove ML)
                if len(V) != 0:
                    vol   = vol   + V            
                else:
                    vol  = vol  + ['NA']
                if len(name[i].text) > 0: # write to csv file
                    d.write(name[i].text + ',' + price[i].text.split(':')[0] + ',' + str(vol[i]) + ',' + body[i] + ',' + tannins[i] + ',' + fruit[i] + ',' + cntry[i].text + '\n')


#Close browser once data are collected
browser.close()









