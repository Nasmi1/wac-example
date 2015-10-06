'''
Created on Feb 16, 2015

@author: casey
'''
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def classify(word, word_classifiers, objects):
    '''
    given a word, the word_classifiers, and the set of objects, this steps through each word in the utterance and each
    object and gets a probability distribution for each object given the results from the word classifiers
    '''
    prob_dists = {}
    if word in word_classifiers: #just ignore words we don't know anything about
        for obj in objects:
            prob_dists[obj] = classify_obj(word, word_classifiers, objects[obj].values())
    return prob_dists

def classify_obj(word, word_classifiers, values):
    values = word_classifiers[word][1].transform(values)
    prob_dist = word_classifiers[word][0].predict_proba(values)
    return prob_dist[0][1]

def train(training_data):
    '''
    The training data is a list of dictionaries and binary class labels (True or False). The dictionary values are the raw values
    the features for the logreg classifier.
    '''
    x = []
    y = []
    for row in training_data:
        x.append(row[0].values())
        y.append(row[1])
    sc = StandardScaler().fit(x)
    x = sc.transform(x)
    regr = linear_model.LogisticRegression(penalty='l1') #
    regr.fit(x, y)
    
    return (regr, sc)