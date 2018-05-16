import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle

college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer.pickle', 'rb') as handle:
    employer = pickle.load(handle)

G = nx.read_gexf("mediumLinkedin.gexf")

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

#print(employer)

lis=[]
col={}
loc={}
emp={}
for i in employer.keys():
    e=employer[i]
    for j in e:
        if 'google' in j : 
            lis.append(i)
            if i in college.keys() :col[i]=college[i]
            if i in location.keys() :loc[i]=location[i]
            if i in employer.keys() :emp[i]=employer[i]

print(col)