# Tech Kuo(thk42) and Ben Wang(bhw44)
# CS 4780 Machine Learning
# 9/18/2014
# Homework 2 Problem 2

import random
import numpy as np
import matplotlib.pyplot as plt
import sys

class Node:
    
	def __init__(self, attr, val, ben, mal, items):
		"""
		precondition: attr, val are the attribute value pair this node will split on.
			ben and mal are the number of benign and malignint tumors respectively.
			items is list of indices of examples to classify.

		"""
		self.attr = attr
		self.val = val
		self.items = items
		self.pos = pos
		self.neg = neg
		self.entropy = self.calculate_entropy()
		self.left = None
		self.right = None

	def calculate_entropy():
		TODO
	


