import math
import matplotlib.pyplot as plt

class Node:
    def __init__(self, x1, x2, is_leaf, label=''):
        self.x1 = x1
        self.x2 = x2
        self.is_leaf = is_leaf
        self.label = label
        self.left = None
        self.right = None

    def classify(self, x1,x2):
        if self.is_leaf:
            return self.label
        if self.x1 != 0: 
            if abs(self.x1 - x1) < .1:
                return self.left.classify(x1,x2)
            else:
                return self.right.classify(x1,x2)
        elif self.x2 != 0:
            if abs(self.x2 - x2) < .1:
                return self.left.classify(x1,x2)
            else:
                return self.right.classify(x1,x2)
        return self.left.classify(x1,x2)



root = Node(3, 0, False)
n1 = Node(0, 0, True, 'blue')
n2 = Node(5, 0, False)
n3 = Node(0, 0, True, 'blue')
n4 = Node(0, 3, False)
n5 =Node(0, 0, True, 'blue')
n6 = Node(0, 0, True, 'red')

root.left = n1
root.right = n2
n2.left = n3
n2.right = n4
n4.left = n5
n4.right = n6

x_range = range(48)
for i in range(len(x_range)):
    x_range[i] /= 8.0
y_range = range(48)
for i in range(len(y_range)):
    y_range[i] /= 8.0

xred = []
yred = []
xblue = []
yblue = []

for x in x_range:
    for y in y_range:
        if root.classify(x, y) == 'red':
            xred.append(x)
            yred.append(y)
        else:
            xblue.append(x)
            yblue.append(y)

plt.plot(xred, yred, 'ro')
plt.plot(xblue, yblue, 'bo')
plt.ylabel('y')
plt.xlabel('x')
plt.show()