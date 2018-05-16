import networkx as nx
import pickle
from collections import Counter

#############################################################################

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
    
#############################################################################

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

def fill(graph,dic):
    
    L=[]
    Nodes = graph.nodes()
    for elt in Nodes:
        if elt not in dic.keys():
            L.append(elt)
    return L

def analyse_and_impute_loc(dic,G,dic_loc):
    
    L=fill(G,dic) #les données manquantes du dictionnaire
    predicted_values={}
    acc=0
    with open('mediumLocation.pickle', 'rb') as handle:
        employer1= pickle.load(handle)
    for i in range(20,45):
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
                        if (n in list(dic_loc.keys())) and len(dic_loc[n])!=0 and dic_loc[n][-1]==dic_loc[nbr][-1]:
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

#############################################################################
    
print("La précision du dictionnaire location est égale à :",analyse_and_impute_loc(location,G,employer)[0])
print("La précision du dictionnaire employer est égale à :",analyse_and_impute_emp(employer,G,location)[0])
print("La précision du dictionnaire college est égale à :",analyse_and_impute_coll(college,G,location)[0])

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
print("Les 5 personnes les plus influentes ont pour identifiant :",l)
