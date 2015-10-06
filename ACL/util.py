'''
Created on Feb 16, 2015

@author: casey
'''
from scipy.spatial import distance
import numpy

def euclidean_distance(a, b):
    return distance.euclidean(a,b)


def print_evaluation_metrics(all_results):
    
    acc_results = []
    rank_results = []
    for results in all_results:
        acc = float(len([r for r in results if r == 1]))
        avg_rank = [1.0 / float(x) for x in results]
        ar = numpy.mean(avg_rank)
        total = float(len(results))
        acc_results.append(acc / total)
        rank_results.append(ar)
        
    print("acc", numpy.mean(acc_results))
    print("std", numpy.std(acc_results))
    print("mrr", numpy.mean(rank_results))
    print("std", numpy.std(rank_results))
    
    
    