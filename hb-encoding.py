# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:46:20 2021

@author: Truffles
"""


import os, pickle, re
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(2019)

import pandas as pd

import torch
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import *

## Need to ensure that pre-processing script is up-to-date import and located in same directory
import pre_processing


def embed_dataset(model, segmented_documents, tokenizer):
    ## doc_sen_embeddings: sentence representations of all documents shape [num of docs, [num of sentences in a docs, [num of tokens in a sent, 768]]]
    case_embeddings = {}
   
    count_empty = 0

    for key, case in tqdm(segmented_documents.items()):
        
        ## Excluding cases that are empty of tokens.
        if len(case) == 0:
            count_empty += 1

        else:
            
            ## doc_sen_embedding: sentence represetations of a document shape [num of sentences in a doc, 768]
            case_embedding = []
            count_exceed = 0
                
            for para in tqdm(case):

                input_ids = tokenizer(para)['input_ids']

                # if number of tokens in a sentence is larger than 256
                if len(input_ids) > 256:
            
                    input_ids = input_ids[:256]
                    count_exceed += 1

                tokens_tensor = torch.tensor([input_ids]).cuda(1)
                encoded_layers = model(tokens_tensor)
                embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

                del encoded_layers
                del tokens_tensor

                case_embedding.append(embeddings_array)
                
            case_embeddings[key] = case_embedding

            if count_exceed > 0:
                print("proportion of paragraphs exceeding max tokenization size: ", count_exceed, " : ", len(case))
            
    print("proportion of empty documents: ", count_empty, " : ", len(case_embeddings))
    
    case_dict = vector_cases(case_embeddings)
    
    return case_dict
    

def gen_embeddings(dataset_name):

    ## Load pre-trained model tokenizer (vocabulary)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    ## Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base')
    
    ## Set the model in evaluation mode to deactivate the DropOut modules
    ## This is IMPORTANT to have reproducible results during evaluation!   
    model.eval()
    model.cuda(1)

    ## Run pre-processing script to obtain relevant tokenization of corpus
    non_seg_docs, vio_seg_docs = pre_processing.process_cases(dataset_name)
    
    ## Generate roberta embeddings of corpus
    non_embed = embed_dataset(model, non_seg_docs, tokenizer)
    vio_embed = embed_dataset(model, vio_seg_docs, tokenizer)

    return non_embed, vio_embed
    
    
def vector_cases(case_embeddings):

    ## case_avg_embeddings: final sentence representations of all documents [num of documents, [num of sentences in a doc, 768]  
    case_dict = {}
    
    for key, case in case_embeddings.items():

        ## temp_doc shape [num of sentences in a doc, 768]
        temp_case = []
        
        for para in case:
            
            avg_para = np.mean(para,axis = 0)
            temp_case.append(avg_para)
        
        case_dict[key] = np.array(temp_case)
        
    return case_dict 


dataset_name = "6"            
print('starting coding hbm')
dict_non, dict_vio = gen_embeddings(dataset_name)

datasets_dirs = os.path.join("datasets", "roberta-base_data")

if not os.path.exists(datasets_dirs):
    os.makedirs(datasets_dirs) 

## Roberta-base encoder used at all times currently. Need to adapt code to permit alternative encodings. 
with open(os.path.join(datasets_dirs, "%s_facts2rel_non.p"%("article" + dataset_name)), 'wb') as fp:
    pickle.dump(dict_non, fp)

with open(os.path.join(datasets_dirs, "%s_facts2rel_vio.p"%("article" + dataset_name)), 'wb') as fp:
    pickle.dump(dict_vio, fp)
