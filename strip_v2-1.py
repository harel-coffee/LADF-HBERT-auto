# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:46:20 2021

@author: Truffles

This script will scrape ECtHR (European Court of Human Rights) cases from the HUDOC website: hudoc.echr.coe.int/
The script is currently fixed to only scrape cases denoted as being available in English (although the website)
is not fully reliable with respect to classifying by language. One can use the scrapecases function to set the
desired article, outcome (violation or nonviolation), and a hard limit on the number of cases to be scraped.
The hard limit can currently be set as any scalar value of 100 and will take the largest number of cases less
than or equal to the hard limit.
"""

import csv, re, requests
import os
os.environ['PYTHONHASHSEED'] = str(2019)

from bs4 import BeautifulSoup
from tqdm import tqdm

"""
def scrapecases is the sole function in this script. Takes as input parameters: the ECHR article, the desired 
case outcome, and the maximum number of cases to be scraped. Outputs two csv files, one for violation cases 
and the other for nonviolation cases. The output files have four columns containing information on the cases: 
id, date, importance (defined by the ECtHR), and the raw html text.
"""
def scrapecases(article, outcome, limit):
    
    assert outcome in ["violation", "nonviolation"]
    
    if outcome == "violation":
        
        query = "violation=" + str(article)
        opposite = "nonviolation=" + str(article)
        filename = "_vio_cases.csv"
        
    elif outcome == "nonviolation":
        
        query = "nonviolation=" + str(article)
        opposite = "violation=" + str(article)
        filename = "_non_cases.csv"
        
    tempidnos = []
    tempdates = []
    tempimps = []
    
    ## start and length are used to iterate through urls in order to scrape for cases in bin sizes of 100 at
    ## a time. Reducing the value of length would allow for finer control of the number of cases scraped.
    ## However, the value of 100 for length suits the current purpose of scraping for article 6 of the ECHR.
    ## Note that this part of the code only finds suitable cases and records their ids so that these can be
    ## used later to scrape the actual case text from the relevant case document url.
    start = 0
    length = 100
    
    ## Iterating through the urls until the hard limit is reached.   
    while start < limit:
            
        url = "https://hudoc.echr.coe.int/app/query/results?query=contentsitename:" + \
        "ECHR%20AND%20(NOT%20(doctype=PR%20OR%20doctype=HFCOMOLD%20OR%20doctype=" + \
        "HECOMOLD))%20AND%20((languageisocode=%22ENG%22))%20AND%20((documentcoll" + \
        "ectionid=%22GRANDCHAMBER%22)%20OR%20(documentcollectionid=%22CHAMBER%22" + \
        "))%20AND%20((" + query + "))%20AND%20(NOT%20(" + opposite + "))&select=" + \
        "sharepointid,Rank,ECHRRanking,languagenumber,itemid,docname,doctype,app" + \
        "lication,appno,conclusion,importance,originatingbody,typedescription,k" + \
        "pdate,kpdateAsText,documentcollectionid,documentcollectionid2,language" + \
        "isocode,extractedappno,isplaceholder,doctypebranch,respondent,advopide" + \
        "ntifier,advopstatus,ecli,appnoparts,sclappnos&sort=&start=" + str(start) + \
        "&length=" + str(length) + "&rankingModelId=11111111-0000-0000-0000-000000000000"
        
        start += length
        
        ## Scraping case ids, dates, and importance values.
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        details = soup.get_text()
        
        regexid = "\d{3}-\d+"
        tempidnos.append(re.findall(regexid, details))
        
        regexdate = '(?<=kpdateAsText\"\:\").*?(?=\s)'
        tempdates.append(re.findall(regexdate, details))

        regeximp = '(?<=importance\"\:\").*?(?=\")'
        tempimps.append(re.findall(regeximp, details))
     
    ## This next part of the court actually produces the output. First a suitable directory is created if
    ## it does not already exist. Then the csv is opened and the details written.
    raw_text_dirs = os.path.join("raw_downloads", "article%s"%(article))
    
    if not os.path.exists(raw_text_dirs):
        os.makedirs(raw_text_dirs)

    with open(os.path.join(raw_text_dirs, "article%s%s"%(article, filename)), 'w', encoding = 'utf-8-sig', newline = '\n') as f:
        
        ## Create the csv writer
        writer = csv.writer(f)
        writer.writerow(["id", "date", "importance", "text"])
        count = 0
        
        idnos = [item for sublist in tempidnos for item in sublist]
        dates = [item for sublist in tempdates for item in sublist]
        imps = [item for sublist in tempimps for item in sublist]
        
        ## Checking that code is functioning correctly as the intitial scraping should have ids, dates, and
        ## importance values for all valid cases.     
        assert len(idnos) == len(dates) == len(imps)
        print("No. of valid cases: ", len(idnos))
        
        ## Using the id for each valid case in order to scrape the raw case text in html format for maximum
        ## flexibility for processing tasks.
        for idx, identity in tqdm(enumerate(idnos)):
            
            url = "https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id=" + identity
            page = requests.get(url)
            case_html = BeautifulSoup(page.content, 'html.parser')
            
            if case_html:
                
                count += 1
                to_write = [identity, dates[idx], imps[idx], case_html]
                writer.writerow(to_write)
                
        print("Total scraped cases: " + str(len(idnos)))
        print("Count of cases in file: " + str(count))

"""
def scrapecases(article, outcome, limit). Where article variable is an integer indicating the article 
(e.g. 6); outcome variable is a string either "violation" or "nonviolation"; and limit variable is an 
integer that denotes the upper limit fornumber of scraped cases as determined by the relevant HUDOC query.
"""
     
article = 6
scrapecases(article, "nonviolation", 1100)
scrapecases(article, "violation", 9100)
