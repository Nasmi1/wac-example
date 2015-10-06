'''
Created on Feb 16, 2015

@author: casey
'''

import sqlite3
import util as util

class TakeCVSqlUtils():

    def __init__(self):
        self.db = "takecv.db"
    
    def set_db(self, db):
        self.db = db
        
    def startdb(self):
        return sqlite3.connect(self.db)
    
    def closedb(self,conn):
        conn.commit()
        conn.close()
        
    def reset(self):
        conn = self.startdb()
        c = conn.cursor()
        c.execute("delete from wrong")
        c.execute("delete from right")
        c.execute("delete from rank")
        self.closedb(conn)
            
    def execute_query(self, query):
        conn = self.startdb()
        c = conn.cursor()
        handle = c.execute(query)
        names = [m for m in map(lambda x: x[0], handle.description)]
        handles = [row for row in handle]        
        results = []
        for handle in handles:
            result = {}
            for col_name,value in zip(names,handle): 
                result[col_name] = value
            results.append(result)
        self.closedb(conn)
        return results
        
    def get_speech(self, eid, source):
        return self.execute_query("select * from {} where episode_id = '{}' order by inc".format(source, eid))
    
    def get_raw_data(self, eid):
        return self.execute_query("select * from cv_piece_raw where episode_id = '{}'".format(eid))
    
    def get_selected_piece(self, eid):
        return self.execute_query("select object from referent where episode_id = '{}'".format(eid))[0]['object'] #should only have 1 result 
    
    def get_landmark_piece(self, eid):
        result = self.execute_query("select object from landmark where episode_id = '{}'".format(eid))
        if result is not None and len(result) > 0:
            return result[0]['object']
        return None
        
    def get_indexed_raw_data(self, eid, index='id'):
        result = self.get_raw_data(eid)
        indexed_results = {}
        for row in result:
            indexed_results[row[index]] = row
        return indexed_results
    
    def get_features(self, data):
        
        for eid in data:
            row = data[eid]
            del row['episode_id']
            del row['position']
            del row['id']
#             row['v_top-skewed'] = 1 if row['v_skew'] == 'top-skewed' else 0
#             row['v_symmetric'] = 1 if row['v_skew'] == 'symmetric' else 0
#             row['v_bottom-skewed'] = 1 if row['v_skew'] == 'bottom-skewed' else 0
#             row['h_top-skewed'] = 1 if row['h_skew'] == 'right-skewed' else 0
#             row['h_symmetric'] = 1 if row['h_skew'] == 'symmetric' else 0
#             row['h_left-skewed'] = 1 if row['h_skew'] == 'left-skewed' else 0
            del row['v_skew']
            del row['h_skew']
            del row['orientation']
            row['c_diff'] = util.euclidean_distance((320,240), (row['pos_x'], row['pos_y'])) # distance from center

        return data
    
    
    def prepare_data(self, speech, source, ignore_landmark=False):
        data = {}
        for utt in speech:
            data_row = {}
            eid = utt['episode_id'] 
            data_row['target'] = self.get_selected_piece(eid)
            if not ignore_landmark:
                data_row['landmark'] = self.get_landmark_piece(eid)
            data_row['speech'] = self.get_speech(eid, source)
            data_row['objects'] = self.get_features(self.get_indexed_raw_data(eid))
            data[eid] = data_row
        return data
    
    def get_all_data(self, source='hand', ignore_landmark=False):
        speech = self.execute_query("select distinct episode_id from {} order by episode_id, inc".format(source))
        return self.prepare_data(speech, source, ignore_landmark)
    
    def get_target_only_data(self, source='hand'):
        speech = self.execute_query("select distinct episode_id from {} where episode_id in (select episode_id from target_episodes) order by episode_id, inc".format(source))
        return self.prepare_data(speech, source)
    
    def get_non_target_only_data(self, source='hand'):
        speech = self.execute_query("select distinct episode_id from {} where episode_id not in (select episode_id from target_episodes) order by episode_id, inc".format(source))
        return self.prepare_data(speech, source)    
    
    def insert_increment(self, eid, inc, word, c_rank, prev_rank):
        conn = self.startdb()
        c = conn.cursor()
        c.execute("insert into rank(episode_id, inc, word, rank, diff) values('{}', {}, '{}', {}, {})".format(eid, inc, word.encode("utf-8"), c_rank, (c_rank - prev_rank)))
        self.closedb(conn)
    
        

