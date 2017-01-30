import sqlite3
import random
import time
import math
import numpy
from sklearn import datasets, linear_model


#### The following cells are for querying the database (take.db should be in the same folder as this notebook) 


def startdb():
    return sqlite3.connect('take.db')
    
def closedb(conn):
    conn.commit()
    conn.close()

def get_raw_data(eid):
    conn = startdb()
    c = conn.cursor()
    result = c.execute('select * from cv_piece_raw where episode_id = ?',(eid,))
    result = [row for row in result] # copy
    closedb(conn)
    return result

def get_words(rows):
    return ' '.join([row[2] for row in rows]).strip()

def get_speech(eid, source='hand'):
    conn = startdb()
    c = conn.cursor()
    result = c.execute('select * from {} where episode_id = ? order by inc'.format(source),(eid,))
    result = [row for row in result] 
    closedb(conn)
    return result


def get_selected_piece(eid):
    conn = startdb()
    c = conn.cursor()
    c.execute('select object from referent where episode_id = ?', (eid,))
    result = c.fetchone()[0]
    closedb(conn)
    return result


#### For extracting features from the database rows

def get_features(sample):
    '''
    A sample consists of raw information retrieved from the database including RGB, HSV, orientation values, as well
    as x,y coordiantes of the object's calcuated centroid, and how it has been skewed.
    
    '''
    #print "sample", sample
    features = {}
    f_name = 'f{}'
    f_ind = 1
    for feature in sample[2:-6]: # rgb, hsv, orientation
        features[f_name.format(str(f_ind))] = feature
        f_ind +=1 
    for feature in sample[-2:]: # x, y
        features[f_name.format(str(f_ind))] = feature
        f_ind +=1 
    features['num_edges'] = sample[11]
    if sample[9] == 'left-skewed': features['h_ls'] = 1.0
    else: features['h_ls'] = 0.0
    if sample[9] == 'right-skewed': features['h_rs'] = 1.0
    else: features['h_rs'] = 0.0
    if sample[9] == 'symmetric': features['h_s'] = 1.0
    else: features['h_s'] = 0.0   
        
    if sample[10] == 'top-skewed': features['v_ts'] = 1.0
    else: features['v_ts'] = 0.0
    if sample[10] == 'bottom-skewed': features['v_bs'] = 1.0
    else: features['v_bs'] = 0.0
    if sample[10] == 'symmetric': features['v_s'] = 1.0
    else: features['v_s'] = 0.0    
    #print "FEATURES", features       
    return features


##### The follwing cells are for training and evaluation

def train(training_data):
    '''
    The training data is a list of dictionaries and class labels (True or False). The dictionary values are the raw values
    the features for the logreg classifier.
    '''
    #print "training_data", training_data[:3]
    x = []
    y = []
    for row in training_data:
        x.append(row[0].values())
        y.append(row[1])
        
    regr = linear_model.LogisticRegression()
    regr.fit(x, y)
    
    return regr


def is_training_data(j, i):
    '''
    returns True if an episode is within a fold's training set
    '''
    return (j < (i-1) * fold_size) or (j > i * fold_size)



# for training data, get the raw features from one sample and max_negs number of negative samples (5 works well)

def process_words(words, eid, utt, max_negs=5):
    '''
    This processes the words for TRAINING. It needs the words dictionary, the episode id, the utterance, and the number of 
    (randomly chosen) negative samples to use for training.
    '''
    selected = get_selected_piece(eid)
    raw_data = get_raw_data(eid)
    random.shuffle(raw_data)
    #grab the positive and negative samples from the set of 15 objects
    neg_samples = []
    pos_sample = None
    for r in raw_data:
        if r[1] == selected: pos_sample = r
        elif len(neg_samples) < max_negs: neg_samples.append(r)
        if len(neg_samples) == max_negs and pos_sample is not None: break
    if pos_sample is None: return
    pos_features = get_features(pos_sample)   
    for word in utt.strip().split():
        if word not in words:
            words[word] = []
        words[word].append((pos_features,True))
        for neg_sample in neg_samples:
            neg_features = get_features(neg_sample)
            words[word].append((neg_features,False))
   


def process_eval_words(objects, eid):
    '''
    the process_words's counterpart for evaluation; this gathers the features for each object
    '''
    raw_data = get_raw_data(eid)
    for row in raw_data:
        objects[row[1]] = get_features(row)
    #print "OBJECTS", objects      
    return objects

def argmax(dist):
    '''
    find the argmax of a distribution in a dictionary
    '''
    m = ('',0)
    for o in dist:
        if dist[o] > m[1]:
            m = (o,dist[o])
    return m[0]

def normalise(prob_dists):
    '''
    Given a dictionary of object ids and probabilities, it normalises the dictionary and returns it 
    '''
    new_dists = {}
    k = sum(prob_dists.values())
    for obj in prob_dists:
        new_dists[obj] = prob_dists[obj] / k
    return new_dists


def classify(utt, word_classifiers, objects):
    '''
    given an utterance, the word_classifiers, and the set of objects, this steps through each word in the utterance and each
    object and gets a probability distribution for each object given the results from the word classifiers
    '''
    word_object_dists = []
    for word in utt.strip().split():
        prob_dists = {}
        if word in word_classifiers: #just ignore words we don't know anything about
            for obj in objects:
                t = numpy.reshape(objects[obj].values(), (1,-1))
                prob_dist = word_classifiers[word].predict_proba(t)
                #prob_dist = word_classifiers[word].prob_classify(objects[obj])
                prob_dists[obj] = prob_dist[0][1]
                #prob_dists[obj] = prob_dist.prob(True)
            prob_dists = normalise(prob_dists)
            word_object_dists.append((word,prob_dists)) #only need the True value, the other can be inferred
    return word_object_dists




'''
     Program begin
'''

conn = startdb()
c = conn.cursor()
res = c.execute('select * from referent order by episode_id') #all episod IDs
eids = [row[0] for row in res]
closedb(conn)

fold_size = 100
num_folds = 10
max_eps = 1000
cor_tot = (0.0,0.0)
entropy = 0.0

source = 'hand' #can switch this to 'asr'

corr_list = []
all_object_dists = {}
all_objects = {}

for i in range(1,num_folds+1):    
        print "processing fold {} out of {}".format(i, num_folds)
        #collect triaining data
        words = {}  # elements of type ({'feature_name':feature_value}:class_label)
        j = 1.0    
        for eid in eids:
            utt = get_words(get_speech(eid,source))
            if j > max_eps: break
            if is_training_data(j, i):
                process_words(words, eid, utt)
            j += 1.0
  

        #train
        word_classifiers = {}
        for word in words:
            word_classifiers[word] = train(words[word]) 
            
        #evaluate    
        j = 1.0
        for eid in eids:
            if j > max_eps: break
            if not is_training_data(j, i):
                utt = get_words(get_speech(eid,source))
                selected = get_selected_piece(eid)
                objects = {}
                process_eval_words(objects, eid)
                word_object_dists = classify(utt, word_classifiers, objects)
                if len(word_object_dists) == 0: continue
                object_dist = word_object_dists[0][1]
                for w in word_object_dists[1:]:
                    for obj in object_dist:
                        object_dist[obj] = object_dist[obj] + w[1][obj]
                object_dist = normalise(object_dist)
    
                guess = argmax(object_dist)
                if guess == selected:
                    cor_tot = (cor_tot[0] + 1.0, cor_tot[1])
                    corr_list.append(eid)
                cor_tot = (cor_tot[0], cor_tot[1]+1.0)
            j += 1.0
    
print "acc", cor_tot, cor_tot[0]/cor_tot[1]






