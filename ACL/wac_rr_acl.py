'''
Created on Feb 16, 2015

@author: casey
'''
from TakeCVSqlUtils  import TakeCVSqlUtils
from nltk.stem.snowball import GermanStemmer
from Distribution import Distribution
import util
import LogRegUtils as logreg
import random
import numpy

 
def get_relational_features(target, landmark):
    t_point = (target['pos_x'],target['pos_y'])
    l_point = (landmark['pos_x'],landmark['pos_y'])
    x_diff = l_point[0] - t_point[0]
    y_diff = l_point[1] - t_point[1]
    distance = util.euclidean_distance(t_point, l_point)
        
    ab = 0 if y_diff < 0 else 1
    lr = 0 if x_diff > 0 else 1
    return {'ab':ab, 'lr':lr,'xdiff':x_diff, 'ydiff': y_diff,'dist':distance}


def add_relation_sample(data, utt, is_negated=False, max_negs=1):  
    global relation_words_list
    
    for word in utt:
    
        target = data['landmark'] if is_negated else data['target']
        landmark = data['target'] if is_negated else data['landmark']
        
        neg_samples = []
        pos_sample = None
        samples = data['objects'].keys()
        
        for r in samples:
            if r == landmark: pos_sample = r
            elif len(neg_samples) < max_negs and r != target: neg_samples.append(r)
            if len(neg_samples) == max_negs and pos_sample is not None: break
        if pos_sample is None: return
        pos_features = get_relational_features(data['objects'][target], data['objects'][pos_sample])
        if word not in relation_words_list:
            relation_words_list[word] = []
        relation_words_list[word].append((pos_features,True))
        for neg_sample in neg_samples:
            neg_features = get_relational_features(data['objects'][target], data['objects'][neg_sample])
            relation_words_list[word].append((neg_features,False))     
    

def add_training_samples(data, utt, selected, max_negs=1):
    '''
    This processes the words for TRAINING. It needs the words dictionary, the episode id, the utterance, and the number of 
    (randomly chosen) negative samples to use for training.
    '''
    global words_list
    #grab the positive and negative samples from the set of objects
    samples = data['objects'].keys()
    samples.remove(selected)
    
    for word in utt:
        if word not in words_list: words_list[word] = []
        words_list[word].append((data['objects'][selected],True))
        for _ in range(0,max_negs):
            words_list[word].append((data['objects'][random.choice(samples)],False))   
            
def add(utt,tag,word):
    global stemmer
    word = stemmer.stem(word)
    
    # some hacky checking to split up some German words
    if type(word) is not list:
        word = word.replace('vom','von')
        word = word.replace("halb", "")
        word = word.replace("dunkel", "")
        word = word.replace("hell", "")
        word = word.replace("mittel", "")
        
    if tag not in utt:
        utt[tag] = []
    
    if type(word) is list:
        for w in word:
            if w == "": continue
            utt[tag].append(w)
    else:
        utt[tag].append(word)
    
            
def prepare_word1(utt, word, tags):
    '''
    word-only model, no structure... i.e., all words are target words
    '''
    add(utt,'t',word)
                
            
def prepare_word(utt, word, tags, r_only=False):
    if 'o' in tags: return
    if 'x' in tags: return
    if 'q' in tags: return
    if 't' in tags and not r_only:
        if 'c' in tags: add(utt,'t',word)
        if 's' in tags: add(utt,'t',word)
        if 'f' in tags: add(utt,'t',word)
    if 'l' in tags and not r_only:
        if len(tags) > 1 and tags[1] == '1':
            pass
        elif len(tags) > 1 and tags[1] == '2':
            pass
        elif len(tags) > 1 and tags[1] == '3':
            pass
        else:        
            if 'c' in tags: add(utt,'l',word)
            if 's' in tags: add(utt,'l',word)
            if 'f' in tags: add(utt,'l',word)
    if 'r' in tags:     
        if 'dar' == word[:3]:
            word = word[3:]
        if 'da' == word[:2]:
            word = word[2:]
        if 'dr' == word[:2]:
            word = word[2:]
      
        if word == '': return
                
        if len(tags) > 1 and tags[1] == '1':
            pass
        elif len(tags) > 1 and tags[1] == '2':
            pass
        elif len(tags) > 1 and tags[1] == '3':
            pass
        else:
            if '-' in tags: add(utt,'r-',word)
            else:  add(utt,'r', word)    
            
def prepare_speech(rows, r_only=False):
    utt = {}
    for row in rows:
#         if not correct: continue
        prepare_word(utt, row['word'], row['tags'])
    return utt

def prepare_training(data, max_negs=1, r_only=False):
    utt = prepare_speech(data['speech'], r_only=r_only)
    target = data['target']
    landmark = data['landmark']
    
    if 't' in utt:
        add_training_samples(data, utt['t'], target, max_negs=max_negs)    
    if 'l' in utt and landmark is not None: # we can consider the landmark for training when we know what it is
        add_training_samples(data,  utt['l'], landmark,  max_negs=max_negs)
        
    if 'r' in utt and landmark is not None:
        add_relation_sample(data, utt['r'], False,  max_negs=max_negs)
    if 'r-' in utt and landmark is not None:
        add_relation_sample(data, utt['r-'], True, max_negs=max_negs)
        
def make_id(t,l):
    return str(t) + '-' + str(l)
        
def apply_relation(target_dist, landmark_dist, relation, negated, objects):
    global relation_word_classifiers
    if relation  not in relation_word_classifiers: 
        relation = "UNK_REL"
        negated = False
        
    combined = Distribution()
    for t in objects:
        for l in objects:
            if t == l: continue
            if negated:
                features = get_relational_features(objects[l], objects[t])
            else:
                features = get_relational_features(objects[t], objects[l])
                
            p = logreg.classify_obj(relation, relation_word_classifiers, features.values())
            combined.add(make_id(t,l), target_dist.get(t) * landmark_dist.get(l) * p)    
            
    combined.marginalise()
    return combined
        
def perform_eval(eid, data):
    global word_classifiers
    global relation_word_classifiers
    global results_outfile
    
    target_dist = Distribution(data['objects'])
    landmark_dist = Distribution(data['objects'])
    target = data['target']
    increment_data = []
    relation = None

    relation_is_negated = False
    inc = 1
    prev_rank = len(data['objects'])
    for w,tags in [(w['word'],w['tags']) for w in data['speech']]:
        utt = {}
        word = None
        c_rank = None
        relation_dist = None
        prepare_word(utt, w, tags)
        objects = data['objects']
        if 't' in utt:
            word = utt['t'][0]
            target_dist.update(logreg.classify(word, word_classifiers, objects))
        if 'l' in utt:
            word = utt['l'][0]
            landmark_dist.update(logreg.classify(word, word_classifiers, objects))
        if 'r' in utt:
            word = utt['r'][0]
            if relation is not None: relation += '_' + word 
            else: relation = word
        if 'r-' in utt:
            word = utt['r-'][0]
            if relation is not None: relation += '_' + word 
            else: relation = word
            relation_is_negated = True            
                
        if relation is not None: # indent this with above for loop to make it incremental
            tdist = target_dist.copy()
            ldist = landmark_dist.copy()
            relation_dist = apply_relation(tdist, ldist, relation, relation_is_negated, objects)
            
    if relation_dist is not None:      
        return relation_dist.rank(target)
    else:
        target_dist.normalise()
        return target_dist.rank(target) 

source = 'asr'
stemmer = GermanStemmer()
sql = TakeCVSqlUtils()
num_folds = 10
data = sql.get_all_data(source)
# data = target_data
print("number of episodes",len(data))

fold_size = len(data) / num_folds

data_keys = data.keys()

results = []
for itr in range(1,2 +1): # evaluate X times and average
    print('iteration', itr)
    iter_results = []
    for i in range(1,num_folds +1): #number of folds
        eval_data = data_keys[i*fold_size:][:fold_size]
        training_data = data_keys[:i*fold_size] + data_keys[(i+1)*fold_size:]
        words_list = {}
        relation_words_list = {}

        # gather training data
        for eid in training_data:
            prepare_training(data[eid], max_negs=2, r_only=True)
            
        # train word classifiers
        word_classifiers = {}
        for word in words_list: 
            word_classifiers[word] = logreg.train(words_list[word])
            
        # train relation classifiers, pipe low-count relations into UNK
        unk_rel = []
        relation_word_classifiers = {}        
        for word in relation_words_list:
            if len(relation_words_list[word]) <= 4: unk_rel += relation_words_list[word]
            else: relation_word_classifiers[word] = logreg.train(relation_words_list[word])
        if len(unk_rel) > 0: relation_word_classifiers['UNK_REL'] = logreg.train(unk_rel)
            
        # evaluate
        for eid in eval_data:
            current_rank = perform_eval(eid, data[eid])
	    iter_results.append(current_rank)
        results.append(iter_results)

util.print_evaluation_metrics(results)

    
    
