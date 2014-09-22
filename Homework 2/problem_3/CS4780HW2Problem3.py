# Tech Kuo(thk42) and Ben Wang(bhw44)
# CS 4780 Machine Learning
# 9/22/2014
# Homework 2 Problem 3

import numpy as np
import sys

def get_prediction_arrays(f):
    data_file = open(f)
    pred_arrays = [[],[]]
    for line in data_file:
        contents = line.split(',')
        for i in range (1,3):
            if i == 1:
                pred_arrays[i-1].append(int(contents[i]))
            else:
                pred_arrays[i-1].append(int(contents[i].strip()))
    return pred_arrays

def get_stats(pa):
    means = [0]*2
    stds = [0]*2
    for i in range(2):
        means[i] = np.mean(pa[i])
        stds[i] = np.std(pa[i])
    return [means,stds]

def get_confidence(stats):
    cis = []
    z95 = 1.96
    for i in range(2):
        lo = stats[i][0] - (z95 * stats[i][1])
        if lo < 0:
            lo = 0
        hi = stats[i][0] + (z95 * stats[i][1])
        if hi > 1:
            hi = 1
        ci = (lo,hi)
        cis.append(ci)
    return cis

pa = get_prediction_arrays(sys.argv[1])
stats = get_stats(pa)
cis = get_confidence(stats)
print "\nConfidence interval for kNN: " + str(cis[0]) + "\n"
print "Confidence interval for Decision Tree: " + str(cis[1]) + "\n"