# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:46:20 2021

@author: Truffles

This script will process the raw html csv files of cases scraped from from the HUDOC website: hudoc.echr.coe.int/
Currently, there are two possible inputs: a violation corpus, and a nonviolation corpus. The script output is in
the form of two pickle file dictionaries, where the key is the case id and the value is the tokenized text. Tokens 
are produced at the fact level, where each bullet (<p>...</p>) is interpreted as a fact, of each case document. 
Currently, relevant text is only taken as pertaining to THE FACTS section of the document. If no significant relevant 
text is found then the case is removed from the processing.
"""

import os, pickle, re
import matplotlib.pyplot as plt

## Need the below to ensure integer markers.
from matplotlib.ticker import MaxNLocator

import numpy as np
np.random.seed(2019)

import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.tokenize import punkt
from bs4 import BeautifulSoup
from tqdm import tqdm


"""
def encode_text takes as input parameters for the data sets: the ECHR article, and the type of outcome (violation or 
nonviolation). It outputs a dictionary (where the key is the case id, and the value is the tokenized text), a count 
of cases containing relevant text, and a count of cases without. 
"""
def encode_text(dataset_name, case_type):

    assert case_type in ["_non_cases.csv", "_vio_cases.csv"]
    
    ## The path must be valid with a csv file specified.
    path = os.path.join("raw_downloads", "article" + dataset_name, "article" + dataset_name + case_type)
    case_data = pd.read_csv(path, encoding = 'utf-8-sig')
    
    """   
    ## These conditionals limit the encodings to cases dating from January 2015 onwards.
    if case_type == "_non_cases.csv":
        case_data = total_data.head(190)

    else:
        case_data = total_data.head(496)


    ## Limit cases to roughly January 2015, but with a balanced data set.
    case_data = total_data.head(200)
    """ 
    segmented_docs = {}
    
    count_rel = 0
    count_irel = 0
    
    for index, case in case_data.iterrows():
        
        segmented_document = []
        
        ## Relevant text is currently taken only from THE FACTS sections. This could be amended if
        ## appropriate.
        #regs_search = ['THE FACTS(.*)THE LAW']
        regs_search = ['THE FACTS(.*)RELEVANT']
        #regs_search = ['RELEVANT(.*)THE LAW']
        relevant = False
        
        for reg_express in regs_search:
    
            passage = re.search(reg_express, str(case["text"]))
            
            ## Only cases containing text in the FACTS section is tokenized and added to the
            ## dictionary. 
            if passage:
            
                relevant = True
                segmented_document = search_text(passage, segmented_document)
                count_rel += 1
            
            else:
                count_irel += 1
        
        if relevant:
            if len(segmented_document) > 0:
                segmented_docs[case["id"]] = segmented_document
            
    return segmented_docs, count_rel, count_irel


## def make_plot takes the dictionary and produces a visualisation of the distribution
## of lengths of the bullets.
def make_plot(segmented_documents, case_type):

    length = []
    
    for i in segmented_documents.values():
        length.append(len(i))
        
    print('max document length: ', np.max(length))
    print('mean document length: ', np.mean(length))
    print('standard deviation: ', np.std(length))

    plt.rcParams.update({'font.size': 20})
    
    ## Need this line to ensure intefer markers.
    ax = plt.figure(figsize=(40,10)).gca()
    
    plt.xlabel('No. of tokens')
    plt.ylabel(case_type)

    plt.xticks(np.arange(0, max(length)+1, 250.0))
    n, bins,patchs = plt.hist(length, 200, facecolor='g', alpha=0.75)
    
    ## Sets y-axis to have integer markers.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

    return


## def process_cases takes as its sole input the ECHR article, and outputs two dictionaries
## - one for violation cases, and the other for nonviolation cases - where the key is the
## case id and the value is the tokenized text.
def process_cases(dataset_name):

    ## Non-violation processing    
    non_seg_docs, non_rel, non_irel = encode_text(dataset_name, "_non_cases.csv")
    
    ## Violation processing    
    vio_seg_docs, vio_rel, vio_irel = encode_text(dataset_name, "_vio_cases.csv")
    
    print("Relevant nonviolation docs: ", non_rel)
    print("Irrelevant nonviolation docs: ", non_irel)   
    make_plot(non_seg_docs, "nonviolation cases")
    
    print("Relevant violation docs: ", vio_rel)
    print("Irrelevant violation docs: ", vio_irel)
    make_plot(vio_seg_docs, "violation cases")
    
    return non_seg_docs, vio_seg_docs


## def search_text takes as input the relevant text pertaining to the regular expression, 
## and the current state of tokenization of the case. It outputs an updated tokenization 
## of the case by tokenizing the input passage.
## 
def search_text(passage, segmented_document):
            
    case_html = BeautifulSoup(str(passage.group(1)), 'html.parser')
    
    ## Uses the html format to segment based on the passages between <p> and </p> markers.
    for para in case_html.find_all("p"):
        
        para_text = para.get_text()
        seg_para = word_tokenize(para_text)
        
        ## Assumes that fully capitalised text are headers and not relevant facts.
        if para_text.isupper():
            pass
        
        ## Assumes that text containing fewer than two tokens is unlikely to be natural
        ## language with useful information.
        elif len(seg_para) > 2:
            
            ## Tokens of this format are likely to be sub-headers rather than information
            ## specific to the case.
            if len(seg_para[0]) == 1 and seg_para[0].isupper():
                pass
            
            ## Removing irrelevant parts of the text from tokenization.
            else:
                
                if seg_para[0].isnumeric():
                    
                    seg_para.pop(0)
                    
                    if seg_para[0] == ".":
                        seg_para.pop(0)
                        
                elif seg_para[0].islower():
                    seg_para.pop(0)
                
                new_text = " ".join(seg_para)
                lower_case = new_text.lower()
                segmented_document.append(lower_case)
    
    return segmented_document

"""
Running the script, taking two input raw html files, one containing case documents pertaining to violations,
and the other for nonviolations. Outputs a dictionary for each file, where the keys are the case ids and the
values are tokenized text from passages deemed relevant for learning.
"""
"""
dataset_name = "6"            
print('starting processing')
dict_non, dict_vio = process_cases(dataset_name)

datasets_dirs = os.path.join("datasets")
    
if not os.path.exists(datasets_dirs):
    os.makedirs(datasets_dirs) 
    
with open(os.path.join(datasets_dirs, "%s_non.p"%("article" + dataset_name)), 'wb') as fp:
    pickle.dump(dict_non, fp)

with open(os.path.join(datasets_dirs, "%s_vio.p"%("article" + dataset_name)), 'wb') as fp:
    pickle.dump(dict_vio, fp)
"""
