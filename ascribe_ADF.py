# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:22:17 2021

@author: Truffles
"""


import pickle
import pandas as pd
import numpy as np
import sys


class AF:    
    def __init__(self, filename):   
        self.defrel = 0
        self.weightvector = []
        self.df = 0
        self.depth = 0
        self.theta = -0.5
        self.filename = filename
        self.readAF()
    
    
    def compdefeat(self, parent):
        
        compdef = False
        parentid = (self.df["FactorNo."][parent])
        
        if parentid[-1] == "c":            
            childid = parentid[:-1]
            
        else:          
            childid = parentid + "c"
            
        if childid in self.df["FactorNo."].values:            
            temp = self.df[self.df["FactorNo."]==childid].index.values
            
            if len(temp) > 1:                
                sys.exit("Problem child value: " + childid + ", cannot have more than one index")
                
            else:                
                compidx = temp[0]
                
            if self.defrel[compidx, parent]:       
                compdef = True
                
        else:          
            compidx = "null"
            
        return compdef, compidx
  
    
    def enumlabtype(self, labtype, violation, baselab):
        
        startlabs = self.genAFlabels(baselab)
        
        self.weightvector = []
        self.weightvector.append([0.0]*self.depth)
        self.weightvector.append([0.0]*self.depth)

        if labtype == "stable":
            
            layers = self.enumstable(startlabs, violation)
            
        return startlabs, layers


    def enumstable(self, labelling, violation):
        
        parentlist = [0]
        layers = []

        if violation:
            self.weightvector[1][0] = 1.0
                
        else:
            self.weightvector[0][0] = 1.0

        elapse = 0
        
        while len(parentlist) > 0:
            
            assert elapse < 50
            layers.append(parentlist)
            children = []
            
            for parent in parentlist:
                
                assert labelling[parent] in ["in", "undec", "out", "fix_undec", "off"]
                
                subchildren = []
                
                for itemidx, item in enumerate(self.defrel[:, parent]):
                    
                    if item <= self.theta:

                        subchildren.append(itemidx)
                        children.append(itemidx)

                self.weightupdate(parent, subchildren, labelling)
                                        
            parentlist = children

            elapse += 1
        
        if elapse > 30:
            print("Elapse in propagating through ADF layers (current limit = 50): " + str(elapse))
        
        return layers    
        

    def genAFlabels(self, annotations):

        newlabs = annotations.copy()
        currentlabs = [99]
        
        elapse = 0
        
        while currentlabs != newlabs and elapse < 200:
            
            currentlabs = newlabs.copy()
            
            for bidx, bval in enumerate(currentlabs):
                
                if bval not in ["fix_undec", "off"]:

                    newlabs[bidx] = "in"
                
                    for aidx, aval in enumerate(currentlabs):
                    
                        assert aval in ["in", "undec", "out", "fix_undec", "off"]
                
                        if self.defrel[aidx][bidx] <= self.theta:
                    
                            if aval == "in":
                        
                                newlabs[bidx] = "out"
                                break
                        
                            elif aval in ["undec", "fix_undec"]:
                                newlabs[bidx] = "undec"
                                     
            elapse += 1
        
        if elapse > 100:       
            print("elapse for genAFlabels (current limit = 200): " + str(elapse))
        
        return newlabs
        
        
    def readAF(self):
        
        self.df = pd.read_csv(self.filename)
        #example = self.df.loc[self.df['FactorNo.'] == "24c", 'Factor'].item()
        
        self.depth = len(self.df['FactorNo.'])
        self.defrel = np.zeros((self.depth, self.depth))
        
        for b in range(self.depth):
            
            strdefeat = self.df['Defeaters'][b]
            
            if isinstance(strdefeat, str):
                
                removebrackets = strdefeat[1:-1]
                defeaters = removebrackets.split()
                
                for defeat in defeaters:
                    
                    a = (self.df[self.df['FactorNo.']==defeat].index.values)[0]
                    self.defrel[a][b] = -1.0
                    
            else:
                
                sys.exit("Problem index: " + str(b) + \
                         ", and problem factor: " + str(self.df['Factor'][b]))
                
        return
    
    
    def weightupdate(self, parent, subchildren, labelling):   
                   
        if len(subchildren) > 0:
            
            compstats = self.compdefeat(parent)
            
            n = len(subchildren)

            if compstats[0]:
                
                # if n == 1 there is no problem since the only child will be
                # the complement which is later set equal to the parentweight.
                delta = (2.0**(n-2) / (2.0**n - 1)) * self.weightvector[0][parent]
                
            else:  
                delta = (2.0**(n-1) / (2.0**n - 1)) * self.weightvector[0][parent]
            
            for subchild in subchildren:

                temp_weight = 1 - (1 - self.weightvector[0][subchild]) * (1 - self.weightvector[1][parent])
                self.weightvector[0][subchild] = temp_weight
                
                temp_weight = 1 - (1 - self.weightvector[1][subchild]) * (1 - delta)
                self.weightvector[1][subchild] = temp_weight

            if subchild == compstats[1]:
                
                self.weightvector[0][compstats[1]] = self.weightvector[1][parent]
                self.weightvector[1][compstats[1]] = self.weightvector[0][parent]
    
        return
    
    
def scaled_weights(neg_weightvector, pos_weightvector):

    neg_generator = [i for i in neg_weightvector if i > 0]
    pos_generator = [i for i in pos_weightvector if i > 0]

    generator = neg_generator + pos_generator
    generator.append(1.0)

    denominator = min(generator)
    
    neg_scaled_weights = np.divide(neg_weightvector, denominator)
    pos_scaled_weights = np.divide(pos_weightvector, denominator)
    
    return neg_scaled_weights, pos_scaled_weights

"""
## Setting the domain for extraction of the relevant ADF.
domain = "article6"
art6 = AF(domain + "_ex_admiss.csv")
base_factors = 32
## Defining the input ascription labellings for the base-level factors
## from the NLP task for the given case.
initlab = ["undec"] * (art6.depth - base_factors) + ["fix_undec"] * base_factors

#print(initlab)

## def enumlabtype(self, labtype, violation, baselab); where labtype = "stable",
## violation in {True, False}, and baselab is a vector indicating the input
## ascription of each node in the scaffolding resulting from the NLP task.
startlabs, layers = art6.enumlabtype("stable", True, initlab)
#print(startlabs, '\n')

## Setting the classification weights for the NLP learning task based on error
## weights from ADF layers.
#neg_class, pos_class = scaled_weights(art6.weightvector[0], art6.weightvector[1])

print(art6.weightvector[0][-base_factors:], '\n')
print(art6.weightvector[1][-base_factors:], '\n')
#print(art6.weightvector[0])
#print(art6.weightvector[1], '\n')

print(startlabs)

#print(neg_class[-35:])
#print(pos_class[-35:], '\n')

for layer in layers:
    
    for item in layer:
        
        if art6.weightvector[0][item] > 0:
            print(art6.df["Factor"][item])
            
    print("")

"""
