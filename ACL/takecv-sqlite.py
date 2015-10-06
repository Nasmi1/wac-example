'''
Created on Jan 22, 2015

@author: casey
'''

import xml.etree.ElementTree as ET
import sqlite3
import string
import os
import pandas as pd
from mumodoIO import *
from ast import literal_eval
from increco import IncReco


try:
    os.remove("takecv.db")
except OSError:
    pass


#needed: ASR, referent ID, landmark ID (if exists), stuff from xml

base_path = '/home/casey/gate/TAKE_CV_Jan15/'
transc = "/home/casey/git/002-takecv-transcriptions/transcriptions/{}_{}.TextGrid"
annotation = base_path + 'DerivedData/takecv-annotation.tsv'
derived = 'DerivedData'
rawdata = 'RawData'

participants = {'r1':'r1_20150120', 'r2':'r2_20150120', 'r3':'r3_20150121', 'r4':'r4_20150121', 'r5':'r5_20150122',
                 'r6':'r6_20150122', 'r7':'r7_20150123', 'r8':'r8_20150123', 'r9':'r9_20150123'}
phases = ['p1','p2']
episodes = 'episodes'


conn = sqlite3.connect('takecv.db')
 
c = conn.cursor()

 
c.execute('CREATE TABLE asr(episode_id text,  inc integer, word text, start_time real, end_time real, tags text, correct integer)')

c.execute('CREATE TABLE hand(episode_id text,  inc integer, word text, start_time real, end_time real, tags text, correct integer)')
 
c.execute('CREATE TABLE referent(episode_id text, object text)')
 
c.execute('CREATE TABLE landmark(episode_id text, object text)')

c.execute('''CREATE TABLE cv_piece
             (episode_id text, id text, color text, top_color text, type text, top_type text, grid text)''')
 
c.execute('CREATE TABLE cv_piece_raw (episode_id text, id text, r double, g double, b double, h double, s double, v double, orientation double, h_skew text, v_skew text, num_edges integer, position text, pos_x integer, pos_y integer)')

c.execute('create table wrong(episode_id text)')
c.execute('create table right(episode_id text)')

c.execute('create table target_episodes(episode_id text)')

c.execute('create table flagged_episodes(episode_id text)')


 
 
# now we attempt to see if we can tag words as being part of a RE from TAKE
take_conn = sqlite3.connect('takecv_mult.db')
take_c = take_conn.cursor()

exclude = set(string.punctuation)

# step through all the data
data = {}
flagged = 0
for r in participants:
    for p in phases: 
        if r == 'r2' and p == 'p2': continue #these are bad, the participant did it wrong
        if r == 'r8' and p == 'p1': continue # there was no audio for this
        print(r, p)
        episode_path = os.path.join(base_path, rawdata, participants[r], p, episodes)
        all_episodes = os.listdir(episode_path)
        
        #get hand-transcription for the entire phase
        try:
            path = transc.format(participants[r],p.replace('p','phase'))            
            hand = open_intervalframe_from_textgrid(path, encoding='utf-16')
        except UnicodeError:
            hand = open_intervalframe_from_textgrid(path, encoding='utf-8')
        except IOError:
            print('not found', path)
            hand = None

        utts = {}
        if hand is not None:
            hand = pd.merge(hand['EPISODES'], hand['A-Utts'], left_index=True, right_index=True)
            for row in hand.iterrows():
                text = row[1]['text_y'].strip()
                ep = row[1]['text_x']
                if text == '': continue
                utts[str(ep)] = text

        for e in all_episodes:
            if 'start_time' in e: continue
            if os.path.exists(os.path.join(episode_path,e, 'flagged.txt')): 
                flagged += 1
                continue # ignore flagged episodes
            if not os.path.exists(os.path.join(episode_path,e,'timestamp.txt')): continue # ignore episodes without timestamps
            pid = r + '.' + e
            #first, insert the ASR
            derived_path = os.path.join(base_path, derived, participants[r], p,'reco',r+p+'.'+str(e)+'.inc_reco')
            if not os.path.exists(derived_path): continue
            inc_reco = IncReco(derived_path).get_last()
            inc  = 1
            for word in inc_reco:
                 s = ''.join(ch for ch in word[2].decode('utf-8') if ch not in exclude)
                 c.execute("INSERT INTO asr VALUES (?,?,?,?,?,?,?)", (pid + '.' + p, inc, s, word[0], word[1],None, None))
                 inc += 1
            #now insert the hand transcription
            inc = 1
            if e in utts:
                for word in utts[e].split():
                    word = word.lower()
                    s = ''.join(ch for ch in word if ch not in exclude)
                    s = s.strip()
                    if s == "": continue
                    c.execute("INSERT INTO hand VALUES (?,?,?,?,?,?,?)", (pid + '.' + p, inc, s, 0, 0,None, None))
                    inc += 1                 
            
            
            #second, get the target and the optional landmark
            ann = open(os.path.join(episode_path, e, 'ann.txt')).readline().strip()
            ann = literal_eval(ann.replace(') (','),(')) # convert the string to a tuple
            
            ref = -1
            if len(ann) == 2: 
                c.execute("INSERT INTO referent VALUES (?,?)", (pid + '.' + p, int(ann[0][0])))
                c.execute("INSERT INTO landmark VALUES (?,?)", (pid + '.' + p, int(ann[1][0])))
                ref = ann[0][0]
            else:
                c.execute("INSERT INTO referent VALUES (?,?)", (pid + '.' + p, int(ann[0])))
                ref = ann[0]            
            
            #third, get the data from the xml
            xml_path = os.path.join(episode_path,e,'setting.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            root = root.findall('timestamp')[0]
            
            l = len(root.findall("object")) 
            ids = ""
            for o in root.findall("object"):
                pos = o.find('position')
                _id = o.attrib['id']
                ids += o.attrib['id']+','
                color = o.find('colour')
                hsv = color.find('hsvValue')
                rgb = color.find('rgbValue')
                if rgb is None: rgb = color.findall('hsvValue')[1] 
                shape = o.find('shape')
                orientation = shape.find('orientation').attrib['value']
                skewness = shape.find('skewness')
                num_edges = shape.find('nbEdges').attrib['value']
                c.execute("INSERT INTO cv_piece_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 
                          (pid + '.' + p, _id, rgb.attrib['R'], rgb.attrib['G'], rgb.attrib['B'],hsv.attrib['H'],hsv.attrib['S'],hsv.attrib['V'],
                           orientation, skewness.attrib['horizontal'], skewness.attrib['vertical'],
                           num_edges,pos.attrib['global'], int(pos.attrib['x']), 480 - int(pos.attrib['y']))) # origin is top left
        
        
                color_dist = color.find('distribution')
                colors = ''
                for attr in color_dist.attrib:
                    colors += attr.lower() + ':' + color_dist.attrib[attr] + ','
                colors = colors[:-1]
                shape_dist = shape.find('distribution')
                shapes = ''
                for attr in shape_dist.attrib:
                    shapes += attr.lower() + ':' + shape_dist.attrib[attr] + ','
                shapes = shapes[:-1]
                c.execute("INSERT INTO cv_piece VALUES (?,?,?,?,?,?,?)", (pid + '.' + p,  _id, colors, color.attrib['BestResponse'].lower(),
                                                                      shapes, shape.attrib['BestResponse'].lower(), pos.attrib['global'].replace(' ','-')))                
                

#now, insert the annotations by updating the asr table
ann = open(annotation).readlines()
ann = [l.strip().split('\t') for l in ann][1:]

for line in ann:
    try:
        c.execute("UPDATE asr SET tags='{}', correct={} WHERE episode_id='{}' and inc={}".format(line[3],line[4], line[0],line[1]))
    except:
        print('Problem inserting ', str(line))      
        
c.execute("insert into target_episodes select episode_id from referent where episode_id not in (select distinct episode_id from asr where tags like '%l%')")
c.execute("insert into flagged_episodes select a.episode_id from (select episode_id, count(*) as c from asr where correct = 0 group by episode_id) a, (select episode_id, max(inc) as m from asr group by episode_id) b where a.episode_id = b.episode_id and a.c > (b.m  / 3)")


conn.commit()
conn.close()
print(flagged)
            
                