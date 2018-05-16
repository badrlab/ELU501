import pickle
import networkx as nx
import numpy as np

G = nx.read_gexf("mediumLinkedin.gexf")
college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)
    

cm=college.copy()

for i in G.nodes():
    if i not in college :
        cm[i]=np.nan

ids=list()
col=set()

for i,j in college.items():
    ids.append(i)
    if isinstance(j,list) :
        col.add(j[0])
    
print(len(G))
l=list(col)

i=list(nx.community.asyn_fluidc(G,4))
j=list(nx.community.asyn_fluidc(G,4))
print(i==j)