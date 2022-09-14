# -*- coding: utf-8 -*-
"""
Created on Fri Dec 3 13:14:35 2021

@author: Truffles
"""


import math
import os
import pickle
from scipy.stats import mannwhitneyu
import statistics
import sys
from optparse import OptionParser


def read_result(model_type, metric, no_epochs, test_prop, max_len):

    save_name = os.path.join("outputs", "article6", model_type + "_epoch%s_prop%s_len%s"%(no_epochs, test_prop, max_len))
    #print("save name: ", save_name)
    sub_dirs = next(os.walk(save_name))[1]
    #print("sub_dirs: ", sub_dirs)
    results = []

    for sub_dir in sub_dirs:

        filepath = os.path.join(save_name, sub_dir, metric)
        #print("filepath: ", filepath)
        with open(filepath, 'rb') as fp:
            results.append(pickle.load(fp))
    
    return results


def calc_acc(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def calc_mcc(tp, tn, fp, fn):

    root = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        
    if root != 0.0:
        mcc = (tp * tn - fp * fn) / math.sqrt(root)
    
    else:
        mcc = 0.0
            
    return mcc


def calc_precision(tp, tn, fp, fn):

    root = tp + fp
        
    if root != 0.0:
        precision = tp / root
    
    else:
        precision = 0.0
            
    return precision


def calc_recall(tp, tn, fp, fn):

    root = tp + fn
        
    if root != 0.0:
        recall = tp / root
    
    else:
        recall = 0.0
            
    return recall


def calc_f1(prec, rec):

    root = prec + rec
        
    if root != 0.0:
        f1 = 2 * prec * rec / root
    
    else:
        f1 = 0.0
            
    return f1
 

def find_best_results(raw_results):

    best_acc = [0.0, "none"]
    best_mcc = [0.0, "none"]
    
    best_vio_prec = [0.0, "none"]
    best_non_prec = [0.0, "none"]
    
    best_vio_rec = [0.0, "none"]
    best_non_rec = [0.0, "none"]
    
    best_vio_f1 = [0.0, "none"]
    best_non_f1 = [0.0, "none"]
    
    for idx, i in enumerate(raw_results):
        
        tp = i["TP"]
        tn = i["TN"]
        fp = i["FP"]
        fn = i["FN"]
        
        ## Processing accuracy.
        acc = calc_acc(tp, tn, fp, fn)
        
        if acc > best_acc[0]:
        
            best_acc[0] = acc
            best_acc[1] = idx
        
        ## Processing mcc score.
        mcc = calc_mcc(tp, tn, fp, fn)
        
        if mcc > best_mcc[0]:
        
            best_mcc[0] = mcc
            best_mcc[1] = idx
        
        ## Processing violation precision.
        vio_prec = calc_precision(tp, tn, fp, fn)
        
        if vio_prec > best_vio_prec[0]:
        
            best_vio_prec[0] = vio_prec
            best_vio_prec[1] = idx
        
        ## Processing no-violation precision.
        non_prec = calc_precision(tn, tp, fn, fp)
        
        if non_prec > best_non_prec[0]:
        
            best_non_prec[0] = non_prec
            best_non_prec[1] = idx
        
        ## Processing violation recall.
        vio_rec = calc_recall(tp, tn, fp, fn)
        
        if vio_rec > best_vio_rec[0]:
        
            best_vio_rec[0] = vio_rec
            best_vio_rec[1] = idx
        
        ## Processing no-violation recall.
        non_rec = calc_recall(tn, tp, fn, fp)
        
        if non_rec > best_non_rec[0]:
        
            best_non_rec[0] = non_rec
            best_non_rec[1] = idx
        
        ## Processing violation F1 score.
        vio_f1 = calc_f1(vio_prec, vio_rec)
        
        if vio_f1 > best_vio_f1[0]:
        
            best_vio_f1[0] = vio_f1
            best_vio_f1[1] = idx
        
        ## Processing no-violation F1 score.
        non_f1 = calc_f1(non_prec, non_rec)
        
        if non_f1 > best_non_f1[0]:
        
            best_non_f1[0] = non_f1
            best_non_f1[1] = idx
    
    return best_acc, best_mcc, best_vio_prec, best_non_prec, best_vio_rec, best_non_rec, best_vio_f1, best_non_f1


def find_iter_results(raw_result, idx):

    print("Chosen index: ", idx)

    tp = raw_result["TP"]
    tn = raw_result["TN"]
    fp = raw_result["FP"]
    fn = raw_result["FN"]
    
    acc = calc_acc(tp, tn, fp, fn)
    mcc = calc_mcc(tp, tn, fp, fn)
    vio_prec = calc_precision(tp, tn, fp, fn)
    non_prec = calc_precision(tn, tp, fn, fp)
    vio_rec = calc_recall(tp, tn, fp, fn)
    non_rec = calc_recall(tn, tp, fn, fp)
    vio_f1 = calc_f1(vio_prec, vio_rec)
    non_f1 = calc_f1(non_prec, non_rec)
    
    return acc, mcc, vio_prec, non_prec, vio_rec, non_rec, vio_f1, non_f1


def get_analysis(raw_results, output_type):

    accuracy = []
    macro_f1 = []
    mcc = []

    for raw_result in raw_results:

        if output_type == "best":

            best_results = find_best_results(raw_result)
            accuracy.append(best_results[0][0])
            mcc.append(best_results[1][0])
            macro = (best_results[6][0] + best_results[7][0]) / 2.0
            macro_f1.append(macro)

        elif output_type == "iteration":

            iter_results = find_iter_results(raw_result[iteration], iteration)
            accuracy.append(best_results[0][0])
            mcc.append(best_results[1][0])
            macro = (iter_results[6][0] + iter_results[7][0]) / 2.0
            macro_f1.append(macro)

    print("Total no. instances: ", len(macro_f1))
    print("accuracies: ", accuracy)
    print("macro f1 scores: ", macro_f1)
    print("mcc scores: ", mcc)

    mean_acc = statistics.mean(accuracy) * 100
    min_acc = min(accuracy) * 100
    max_acc = max(accuracy) * 100

    mean_macro_f1 = statistics.mean(macro_f1) * 100
    min_macro_f1 = min(macro_f1) * 100
    max_macro_f1 = max(macro_f1) * 100

    mean_mcc = statistics.mean(mcc) * 100
    min_mcc = min(mcc) * 100
    max_mcc = max(mcc) * 100

    print("average accuracy: ", mean_acc)
    print("accuracy range: -", mean_acc - min_acc, ", +", max_acc - mean_acc)

    print("average macro f1: ", mean_macro_f1)
    print("macro_f1 range: -", mean_macro_f1 - min_macro_f1, ", +", max_macro_f1 - mean_macro_f1)

    print("average mcc: ", mean_mcc)
    print("mcc range: -", mean_mcc - min_mcc, ", +", max_mcc - mean_mcc)
    print("\n")

    return accuracy, macro_f1, mcc


if __name__ == "__main__":
    
    ## Loading the data
    parser = OptionParser(usage = 'usage: -i iteration -o output_type')
    
    parser.add_option("-i", "--iteration", action = "store", type = "int", dest = "iteration", help = "iteration for specific results", default = 0)
    parser.add_option("-o", "--output_type", action = "store", type = "string", dest = "output_type", help = "best or iteration", default = "best")

    (options, _) = parser.parse_args()
    
    iteration = options.iteration
    output_type = options.output_type
    
    ## Running the analysis
    hybrid_results = read_result("hybrid_v4.3", "test_confusion.p", "30", "0.2", "256")
    hbert_results = read_result("hbert", "test_confusion.p", "30", "0.2", "256")

    print("Hybrid Analysis...")
    hybrid_accuracy, hybrid_macro_f1, hybrid_mcc = get_analysis(hybrid_results, output_type)
    print("Hbert Analysis...")
    hbert_accuracy, hbert_macro_f1, hbert_mcc = get_analysis(hbert_results, output_type)

    print("Accuracy p value: ", mannwhitneyu(hybrid_accuracy, hbert_accuracy)[1])
    print("Macro F1 Score p value: ", mannwhitneyu(hybrid_macro_f1, hbert_macro_f1)[1])
    print("MCC Score p value: ", mannwhitneyu(hybrid_mcc, hbert_mcc)[1])
