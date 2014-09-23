# Tech Kuo(thk42) and Ben Wang(bhw44)
# CS 4780 Machine Learning
# 9/18/2014
# Homework 2 Problem 2

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

class Node:

    num_nodes = 0
    num_leaves = 0
    tree_depth = 0
    max_depth = 0
    num_nodes_per_level = [0]*20
    num_leaves_per_level = [0]*20

    def __init__(self, ben, mal, attr_range, items, entropy, depth):
        """
        precondition:
            ben and mal are the number of benign and malignint tumors respectively.
            attr_range is a list of tuples representing the lower,upper bound of each attribute.
            items is list of indices of examples to classify.
            entropy is the entropy of the instances represented by items.
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
        self.depth = depth

        if ben > mal:
            self.tree_label = 'B'
        else:
            self.tree_label = 'M'
        
        Node.tree_depth = max(Node.tree_depth, self.depth)
        Node.num_nodes += 1
        if self.is_leaf():
            Node.num_leaves += 1
            Node.num_leaves_per_level[self.depth-1] += 1
        Node.num_nodes_per_level[self.depth-1] += 1

    def __str__(self):
        s = "Attribute " + str(self.attr) + " <= " + str(self.val) + "\n" 
        s += "Entropy: " + str(self.entropy) + "\n"
        s += "[" + str(self.ben) + ", " + str(self.mal) + "] \n"
        s += "Depth of Node: " + str(self.depth) + "\n"
        s +=  "Tree Label: " + self.tree_label + "\n"
        return s

    @staticmethod
    def calc_entropy(b, m):
        """
        precondition: b and m are the count of different types in a collection.
        return: the entropy of this collection.
        """

        if b == 0 or m == 0:
            return 0
        p_plus = b/float(b + m)
        p_minus = m/float(b + m)
        return -(p_plus * np.log2(p_plus) + p_minus * np.log2(p_minus))

    def is_leaf(self):
        """
        return: this Node is a leaf.
        """
        return (self.depth == Node.max_depth) or (self.ben == 0) or (self.mal == 0)

    def find_best_attribute(self):
        """
        summary: Iterates through all possible attribute, value pairs and finds the best split.
            Records the best attribute, value to split on in the fields of this Node

        return: The values associated with splitting. Will be a list l.
            l[0] = # of benign examples in the left subset
            l[1] = # of malignint examples in the left subset
            l[2] = a list of indices indicating the examples in the left subset
            l[3] = entropy of the left subset
            l[4] = # of benign examples in the right subset
            l[5] = # of malignint examples in the right subset
            l[6] = a list of indices indicating the examples in the right subset
            l[7] = entropy of the right subset
        """
        ben_lte = 0 #-1 indicates a leaf node
        mal_lte = 0 #-1 indicates a leaf node
        ben_gt = 0 #-1 indicates a leaf node
        mal_gt = 0 #-1 indicates a leaf node

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
            for val in range(lo, hi+1):
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
        """
        summary: Helper function for find_best_attribute.
        """
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
        """
        summary: Builds the decision tree rooted at this Node. 
            Uses the id3 algorithm to find best splitting criterion.
        """
        if not self.is_leaf():
            rv = self.find_best_attribute()
            lte_attr_range = copy.deepcopy(self.attr_range)
            lte_attr_range[self.attr][1] = self.val
            gt_attr_range = copy.deepcopy(self.attr_range)
            gt_attr_range[self.attr][0] = self.val + 1
            self.left = Node(rv[0], rv[1], lte_attr_range, rv[2], rv[3], self.depth + 1)
            self.right = Node(rv[4], rv[5], gt_attr_range, rv[6], rv[7], self.depth + 1)
            self.left.make_babies()
            self.right.make_babies()
    
    def classify(self, item):
        """
        precondition: item is the instance to be classified. 
        returns: 'B' if item is classified benign, 'M' if item is classified malignint.
        """
        if self.is_leaf():
            return self.tree_label
        if item[self.attr] <= self.val:
            return self.left.classify(item)
        else:
            return self.right.classify(item)

def init_tree(items):
    """
    precondition: items is a list of indices of instances.
    return: Root node of decision tree built by training on items.
    """
    root_ben = 0
    root_mal = 0
    for example in items:
        if example[0] == 'B':
            root_ben += 1
        else:
            root_mal += 1
    root_entr = Node.calc_entropy(root_ben, root_mal)
    root_items = range(len(items))
    root_attr_range = [[1,10] for x in range(10)]
    root_depth = 1
    root = Node(root_ben, root_mal, root_attr_range, root_items, root_entr, root_depth)
    print root 
    root.make_babies()
    return root 

def classify_items(dt, items):
    """
    precondition: dt is a decision tree, items is a list of instances.
    summary: Modifies each instance by appending the label chosen by dt.
    """
    for example in items:
        tree_label = dt.classify(example)
        example[10] = tree_label

def find_accuracy(classified_items):
    correct = 0
    for item in classified_items:
        if item[0] == item[10]:
            correct += 1
    return float(correct) / len(classified_items)


def parse_file(f):
    data_file = open(f)
    l = []
    for line in data_file:
        contents = line.split(' ')
        for i in range(1,10):
            temp = contents[i].split(':')
            contents[i] = int(temp[1])
        item = contents[:10]
        item.append('N')
        l.append(item)
        # last location will be overwritten when instance is classified

    return l

def print_root_and_children(dt):
    print "Root:\n" + str(dt)
    print "Left Child:\n" + str(dt.left)
    print "Right Child:\n" + str(dt.right)

train_set = parse_file(sys.argv[1])
val_set = parse_file(sys.argv[2])
test_set = parse_file(sys.argv[3])
decision_tree = init_tree(train_set)

classify_items(decision_tree, train_set)
classify_items(decision_tree, test_set)
acc_train = find_accuracy(train_set)
acc_test = find_accuracy(test_set)
print "-----------Decision Tree----------- \n"
print_root_and_children(decision_tree)
print "Total Number of Nodes: " + str(Node.num_nodes)
print "Total Number of Leaves: " + str(Node.num_leaves)
print "Accuracy of training set: " + str(acc_train)
print "Accuracy of test set: " + str(acc_test)
print "Depth of tree is: " + str(Node.tree_depth)


max_depths = range(2,21)
val_acc = []
test_acc = []

print ''
print 'Depth | Leaves | Nodes'
print '----------------------'
for depth in max_depths:
    Node.max_depth = depth
    leaves = sum(Node.num_leaves_per_level[:depth-1]) + Node.num_nodes_per_level[depth-1]
    nodes = sum(Node.num_nodes_per_level[:depth])
    print '  {0:2d}  |   {1:2d}   |  {2:2d}  '.format(depth, leaves, nodes)
    classify_items(decision_tree,val_set)
    classify_items(decision_tree, test_set)
    val_acc.append(find_accuracy(val_set))
    test_acc.append(find_accuracy(test_set))



print 'test accuracies: ' + str(test_acc)
#print 'validation accuracies: ' + str(val_acc)
plt.plot(max_depths, test_acc, 'bo')
plt.xlabel('maximum depth')
plt.ylabel('accuracies')
plt.show()
