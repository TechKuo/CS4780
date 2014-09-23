import math
import matplotlib.pyplot as plt


train_x = [1,1,2,2,4,6,3,3,4,5,5,6]
train_y = [4,6,1,2,1,1,3,4,3,5,6,5]
train_color = ['red','red','red','red','red','red','blue','blue','blue','blue','blue','blue']

training_set = []

xblue = []
yblue = []

xred = []
yred = []


class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def distance(self, x, y):
        return math.sqrt((self.x-x) **2 + (self.y-y) **2)

def find_nearest_neighbors(x, y, k):
    def cmp(p1, p2):
        diff = p1.distance(x, y) - p2.distance(x, y)
        if diff < 0:
            return -1
        elif diff == 0:
            return 0
        else:
            return 1
    sorted_points = sorted(training_set, cmp)
    return sorted_points[:k]

def store_point(x, y, neighbors):
    red = 0
    blue = 0
    for point in neighbors:
        if point.color == "blue":
            blue += 1
        else:
            red += 1
    if red > blue:
        xred.append(x)
        yred.append(y)
    else:
        xblue.append(x)
        yblue.append(y)


for i in range(12):
    new_point = Point(train_x[i], train_y[i], train_color[i])
    training_set.append(new_point)
x_range = range(48)
for i in range(len(x_range)):
    x_range[i] /= 8.0
y_range = range(48)
for i in range(len(y_range)):
    y_range[i] /= 8.0

for x in x_range:
    for y in y_range:
        neighbors = find_nearest_neighbors(x, y, 1)
        store_point(x, y, neighbors)
plt.plot(xred, yred, 'ro')
plt.plot(xblue, yblue, 'bo')
plt.ylabel('y')
plt.xlabel('x')
plt.show()