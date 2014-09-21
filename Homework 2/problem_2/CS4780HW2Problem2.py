# Tech Kuo(thk42) and Ben Wang(bhw44)
# CS 4780 Machine Learning
# 9/18/2014
# Homework 2 Problem 2

import random
import numpy as np
import sys
import copy


class Node:

    def __init__(self, ben, mal, attr_range, items, entropy):
        """
        precondition: attr, val are the attribute value pair this node will split on.
            ben and mal are the number of benign and malignint tumors respectively.
            items is list of indices of examples to classify.

        """
        self.attr = -1 # will be assigned in find_best_attribute
        self.val = -1 # will be assigned in find_best_attribute
        self.items = items
        self.ben = ben
        self.mal = mal
        self.attr_range = attr_range
        self.entropy = entropy
        self.left = None
        self.right = None

    def __str__(self):
        s = "Attribute " + str(self.attr) + ", <= " + str(self.val) + "\n" 
        s += "Entropy: " + str(self.entropy) + "\n"
        s += "[" + str(self.ben) + ", " + str(self.mal) + "] \n"
        return s

    @staticmethod
    def calc_entropy(b, m):
        if b == 0 or m == 0:
            return 0
        p_plus = b/float(b + m)
        p_minus = m/float(b + m)
        return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))

    def is_pure(self):
        return (self.ben == 0) or (self.mal == 0)

    def find_best_attribute(self):
        ben_lte = 0
        mal_lte = 0
        ben_gt = 0
        mal_gt = 0

        max_ben_lte = -1
        max_mal_lte = -1
        max_ben_gt = -1
        max_mal_gt = -1
        max_entropy_lte = -1
        max_entropy_gt = -1
        max_gain = -1
        max_lte_items = None
        max_gt_items = None
        max_attr = -1
        max_val = -1

        curr_lte_items = []
        curr_gt_items = []

        for attr in range(1,10):
            lo = self.attr_range[attr][0]
            hi = self.attr_range[attr][1]
            for val in range(lo, hi):
                rv = self.calc_values(attr, val)
                ben_lte, mal_lte, curr_lte_items, entropy_lte = rv[:4]
                ben_gt, mal_gt, curr_gt_items, entropy_gt, gain = rv[4:]

                # if this is a better split, store this attr, val pair
                if gain > max_gain:
                    max_ben_lte = ben_lte
                    max_mal_lte = mal_lte
                    max_lte_items = curr_lte_items
                    max_entropy_lte = entropy_lte

                    max_ben_gt = ben_gt
                    max_mal_gt = mal_gt
                    max_gt_items = curr_gt_items
                    max_entropy_gt = entropy_gt
                    max_gain = gain
                    
                    max_attr = attr
                    max_val = val

        rv = [max_ben_lte, max_mal_lte, max_lte_items, max_entropy_lte]
        rv.extend([max_ben_gt, max_mal_gt, max_gt_items, max_entropy_gt])
        self.attr = max_attr
        self.val = max_val
        return rv

    def calc_values(self, attr, val):
        ben_lte = 0
        mal_lte = 0
        ben_gt = 0
        mal_gt = 0

        curr_lte_items = []
        curr_gt_items = []

        for item in self.items:
            if train_set[item][attr] <= val:
                curr_lte_items.append(item)
                if train_set[item][0] == 'B':
                    ben_lte += 1
                else:
                    mal_lte += 1
            else:
                curr_gt_items.append(item)
                if train_set[item][0] == 'B':
                    ben_gt += 1
                else:
                    mal_gt += 1
        entropy_lte = Node.calc_entropy(ben_lte, mal_lte)
        entropy_gt = Node.calc_entropy(ben_gt, mal_gt)
        gain = self.entropy - len(curr_lte_items)/float(len(self.items))*entropy_lte
        gain -= (len(curr_gt_items)/float(len(self.items)))*entropy_gt

        return ben_lte, mal_lte, curr_lte_items, entropy_lte, ben_gt, mal_gt, curr_gt_items, entropy_gt, gain
    
    def make_babies(self):
        if not self.is_pure():
            rv = self.find_best_attribute()
            lte_attr_range = copy.deepcopy(self.attr_range)
            lte_attr_range[self.attr][1] = self.val
            gt_attr_range = copy.deepcopy(self.attr_range)
            gt_attr_range[self.attr][0] = self.val + 1
            self.left = Node(rv[0], rv[1], lte_attr_range, rv[2], rv[3])
            self.right = Node(rv[4], rv[5], gt_attr_range, rv[6], rv[7])
            print self
            self.left.make_babies()
            self.right.make_babies()

def init_tree():
    root_ben = 0
    root_mal = 0
    for example in train_set:
        if example[0] == 'B':
            root_ben += 1
        else:
            root_mal += 1
    root_entropy = Node.calc_entropy(root_ben, root_mal)
    root_items = range(len(train_set))
    root_attr_range = [[1,10] for x in range(10)]
    root = Node(root_ben, root_mal, root_attr_range, root_items, root_entropy)
    decision_tree = root.make_babies()


def parse_file(f):
    data_file = open(f)
    l = []
    for line in data_file:
        contents = line.split(' ')
        for i in range(1,10):
            temp = contents[i].split(':')
            contents[i] = int(temp[1])
        l.append(tuple(contents[:10]))
    return l

train_set = parse_file(sys.argv[1])
init_tree()


