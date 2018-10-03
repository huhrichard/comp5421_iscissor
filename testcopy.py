class test:
    def __init__(self, data):
        self.data = data
l = [1, 2, 3, 4]
t = test(l)
import copy
a = copy.copy(t)
b = copy.deepcopy(t)
l.append(5)
print a.data
print b.data