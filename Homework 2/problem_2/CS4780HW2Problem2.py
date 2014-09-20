# Tech Kuo(thk42) and Ben Wang(bhw44)
# CS 4780 Machine Learning
# 9/18/2014
# Homework 2 Problem 2

import random
import numpy as np
import matplotlib.pyplot as plt
import sys

class Node:
    
	def __init__(self, attr, val, ben, mal, attr_range, entropy, items):
		"""
		precondition: attr, val are the attribute value pair this node will split on.
			ben and mal are the number of benign and malignint tumors respectively.
			items is list of indices of examples to classify.

		"""
		self.attr = attr
		self.val = val
		self.items = items
		self.ben = ben
		self.mal = mal
		self.attr_range = attr_range
		self.entropy = entropy 
		self.left = None
		self.right = None

	def __str__(self):
		return "Attribute " + self.attr + "<= " + self.val + "\n" + "[" + 
				self.ben + ", " + self.mal + "]"

	def calculate_entropy(b,m):
		p_plus = (b/b + m))
		p_minus = (m/(b + m))
		return -(p_plus * np.log2(p_plus) - p_minus * np.log2(p_minus))

	def is_pure(self):
		return (self.ben == 0) or (self.mal == 0)


