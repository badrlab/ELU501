import networkx as nx
import pickle
from collections import Counter

def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[]
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values
    
 
def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])/len(groundtruth[p_key])        
        # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction/len(pred)
   

# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
#print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
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

'''print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes=[]
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))


# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'
employer_predictions=naive_method(G, empty_nodes, employer)
groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle:
    groundtruth_employer = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)
print("%f%% of the predictions are true" % result)
print("Very poor result!!! Try to better!!!!")'''

# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures

def fill(graph,dic):
    
    L=[]
    Nodes = graph.nodes()
    for elt in Nodes:
        if elt not in dic.keys():
            L.append(elt)
    return L

global elt1

#############################################################################

def analyse_and_impute_loc(dic,G,dic_loc):
    
    L=fill(G,dic) #les données manquantes du dictionnaire
    predicted_values={}
    acc=0
    with open('mediumLocation.pickle', 'rb') as handle:
        employer1= pickle.load(handle)
    for i in range(5,20):
        list_commun=list(nx.community.asyn_fluidc(G,i))
        for elt in list_commun:
            list_elt=list(elt)
            data_to_impute=[]
            for dat in L:
                if dat in list_elt:
                    data_to_impute.append(dat) 
            for n in data_to_impute:
                nbrs_attr_values=[]
                for nbr in list_elt:
                    if nbr in dic:
                        for val in dic[nbr]:
                            nbrs_attr_values.append(val)
                        if nbr in G.neighbors(n):
                            for val in dic[nbr]:
                                nbrs_attr_values.append(val)
                        if (n in list(dic_loc.keys())) and dic_loc[n][-1]==dic_loc[nbr][-1]:
                            nbrs_attr_values.append(dic[nbr][-1])
                            nbrs_attr_values.append(dic[nbr][-1])

                predicted_values[n]=[]
                if nbrs_attr_values: # non empty list
                    # count the number of occurrence each value and returns a dict
                    cpt=Counter(nbrs_attr_values)
                    # take the most represented attribute value among neighbors
                    a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
                    predicted_values[n].append(a)
                    
                    
        for elt in list(dic.keys()):
            predicted_values[elt]=dic[elt]
        if evaluation_accuracy(employer1, predicted_values)>=acc:
            acc=evaluation_accuracy(employer1, predicted_values)
    return [100*acc,predicted_values]

#############################################################################
    
def analyse_and_impute_emp(dic,G,dic_loc):
    
    L=fill(G,dic) #les données manquantes du dictionnaire
    predicted_values={}
    acc=0
    with open('mediumEmployer.pickle', 'rb') as handle:
        employer1= pickle.load(handle)
    for i in range(5,20):
        list_commun=list(nx.community.asyn_fluidc(G,i))
        for elt in list_commun:
            list_elt=list(elt)
            data_to_impute=[]
            for dat in L:
                if dat in list_elt:
                    data_to_impute.append(dat) 
            for n in data_to_impute:
                nbrs_attr_values=[]
                for nbr in list_elt:
                    if nbr in dic:
                        for val in dic[nbr]:
                            nbrs_attr_values.append(dic[nbr][-1])
                        if nbr in G.neighbors(n):
                            for val in dic[nbr]:
                                nbrs_attr_values.append(dic[nbr][-1])
                        if (n in list(dic_loc.keys())) and dic_loc[n][-1]==dic_loc[nbr][-1]:
                            for val in dic[nbr]:
                                nbrs_attr_values.append(val)
                                nbrs_attr_values.append(val)
                                nbrs_attr_values.append(val)
                                nbrs_attr_values.append(val)
                predicted_values[n]=[]
                if nbrs_attr_values: # non empty list
                    # count the number of occurrence each value and returns a dict
                    cpt=Counter(nbrs_attr_values)
                    # take the most represented attribute value among neighbors
                    a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
                    predicted_values[n].append(a)
                    
        for elt in list(dic.keys()):
            predicted_values[elt]=dic[elt]
        if evaluation_accuracy(employer1, predicted_values)>=acc:
            acc=evaluation_accuracy(employer1, predicted_values)
    return [100*acc,predicted_values]

#############################################################################

def analyse_and_impute_coll(dic,G,dic_loc):
    
    L=fill(G,dic) #les données manquantes du dictionnaire
    predicted_values={}
    acc=0
    with open('mediumCollege.pickle', 'rb') as handle:
        employer1= pickle.load(handle)
    for i in range(5,20):
        list_commun=list(nx.community.asyn_fluidc(G,i))
        for elt in list_commun:
            list_elt=list(elt)
            data_to_impute=[]
            for dat in L:
                if dat in list_elt:
                    data_to_impute.append(dat) 
            for n in data_to_impute:
                nbrs_attr_values=[]
                for nbr in list_elt:
                    if nbr in dic:
                        for val in dic[nbr]:
                            nbrs_attr_values.append(dic[nbr][-1])
                        if nbr in G.neighbors(n):
                            for val in dic[nbr]:
                                nbrs_attr_values.append(dic[nbr][-1])
                        if (n in list(dic_loc.keys())) and dic_loc[n][-1]==dic_loc[nbr][-1]:
                            for val in dic[nbr]:
                                nbrs_attr_values.append(val)
                                nbrs_attr_values.append(val)
                predicted_values[n]=[]
                if nbrs_attr_values: # non empty list
                    # count the number of occurrence each value and returns a dict
                    cpt=Counter(nbrs_attr_values)
                    # take the most represented attribute value among neighbors
                    a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
                    predicted_values[n].append(a)
                    
        for elt in list(dic.keys()):
            predicted_values[elt]=dic[elt]
        if evaluation_accuracy(employer1, predicted_values)>=acc:
            acc=evaluation_accuracy(employer1, predicted_values)
    return [100*acc,predicted_values]

print(analyse_and_impute_loc(location,G,employer)[0])
predicted_location=analyse_and_impute_loc(location,G,employer)[1]
nodes=nx.eigenvector_centrality(G)
L={}#L is a dict we re gonna fill with people who live in san francisco bay area

for i in nodes.keys():
    if predicted_location[i]==['san francisco bay area']:
        L[i]=nodes[i]
        
V=L.copy()
l=[]
n=0
while n<5:
    l.append(max(L))
    L.pop(l[-1])
    n+=1    
print(l)