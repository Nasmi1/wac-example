'''
Created on Feb 16, 2015

@author: casey
'''
import operator

class Distribution():
    
    def __init__(self, objects={}):
        self.dist = self.make_uniform_dist(objects)
        self.uniform = True
        
    def set(self, other):
        self.dist = other.dist
        self.uniform = other.is_uniform()
        
    def is_uniform(self):
        return self.uniform
    
    def copy(self):
        copy = self.dist.copy()
        temp = Distribution()
        temp.dist = copy
        temp.uniform = self.is_uniform()
        return temp
    
    def make_uniform_dist(self, objects):
        dist = {}
        size = len(objects)
        for o in objects:
            dist[o] = 1.0 / size
        return dist
    
    def add(self, name, prob):
        self.uniform = False
        self.dist[name] = prob
        
    def get_prob(self, name):
        return self.dist[name]
        
    def update(self, new_dist):
        for obj in new_dist:
            if obj not in self.dist: self.dist[obj] = 0.0
            self.dist[obj] = self.dist[obj] + new_dist[obj]
        self.uniform = False
            
    def order_by_prob(self,top=-1):
        sorted_obj = sorted(self.dist.items(), key=operator.itemgetter(1))
        sorted_obj.reverse()
        if top == -1:
            return sorted_obj
        else: 
            return sorted_obj[:top]            
            
    def normalise(self):
        k = sum([o for o in self.dist.values()])
        for obj in self.dist:
            self.dist[obj] = self.dist[obj] / k
        
    def size(self):
        return len(self.dist)
    
    def limit(self, amount):
        dist = self.order_by_prob(amount)
        new_dist = {}
        for row in dist:
            new_dist[row[0]] = row[1]
        self.dist = new_dist
    
    def get(self, i):
        return self.dist[i]
    
    def rank(self, selected):
        ind = 1
        for s in self.order_by_prob():
            if '-' in s[0]: s = s[0].split('-')
            if s[0] == selected: return ind
            ind += 1
        return len(self.dist)
    
    def marginalise(self):
        new_dist = {}
        for s in self.dist:
            p = self.dist[s]
            s = s.split('-')[0]
            if s not in new_dist: new_dist[s] = 0.0
            new_dist[s] += p
        self.dist = new_dist
#         self.normalise()
        
    def __str__(self):
        return str(self.dist)
            
    
        
    
    