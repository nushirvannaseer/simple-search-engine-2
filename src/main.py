# Imports
import math
import numpy as np
from typing import Counter
import argparse
import pickle
import copy
import os
import threading
import logging


# Constants
TERM_IDS = '../data/termids.txt'
TERM_INFO = '../data/term_info.txt'
TERM_INDEX = '../data/term_index.txt'
DOC_INDEX = '../data/doc_index.txt'
DOC_IDS = '../data/docids.txt'
QUERIES = '../data/queries.txt'
QUERY_IDS = '../data/query_ids.txt'
DOC_LEN_PATH = '../data/doc_len_dict.pkl'

QUERY_IDS_LIST = []
LIST_OF_QUERIES = []
TOTAL_DOC_IDS = 0
DOC_NAMES = []
DOC_ID_TF = {}
DF_DICT ={}
DOC_LEN = {}
AVG_LEN = 1171
TERMID_DICT = {}

f = open(TERM_INFO, 'r')
li= list(f)
for line in li:
    line = line.split('\t')
    doc_id = int(line[0])
    df = int(line[3])
    DF_DICT[doc_id] = df
f.close()

f = open(QUERIES, 'r')
LIST_OF_QUERIES = list(f)
f.close()

docids = open(DOC_IDS, 'r')
TOTAL_DOC_IDS = len(list(docids))
docids.close()

docids = open(DOC_IDS, 'r')
DOC_NAMES = list(docids)
docids.close()

f = open(QUERIES, 'r')
list_of_queries = list(f)
f.close()

with open(QUERY_IDS, 'r') as q_file:
    QUERY_IDS_LIST = list(q_file)

for i in range(len(QUERY_IDS_LIST)):
    QUERY_IDS_LIST[i] = int(QUERY_IDS_LIST[i])
    
def findLen(d):
  global DOC_LEN
  f= open(DOC_INDEX, 'r')
  li = list(f)
  length = 0
  for line in li:
    if int(line.split('\t')[0]) == d+1:
      # return length
      break
    if int(line.split('\t')[0]) == d:
      length += int(line.split('\t')[2])
  DOC_LEN[int(d)] = length



def findAvgLen(D):
  length = 0
  for d in range(D):
    if d in DOC_ID_TF.keys():
      length += DOC_LEN[d]
  return length/D

# AVG_LEN = findAvgLen(TOTAL_DOC_IDS)
sumLenD = AVG_LEN * TOTAL_DOC_IDS

def termid_dict():
  f = open(TERM_IDS, 'r');
  li = list(f)
  dic = {i.split('\t')[1].split('\n')[0] : i.split('\t')[0] for i in li}
  return(dic)

# d = int documentID
# i = int termID
def tf(d, i):
  f = open(TERM_INFO, 'r')
  li = list(f)
  f2 = open(TERM_INDEX, 'r')
  f2.seek(int(li[i].split('\t')[1]))
  term_info = f2.readline().split('\t')
  for doc in range(1, len(term_info)):
    if int(term_info[doc].split(':')[0]) == d:
      return int(term_info[doc].split(':')[1])
  return 0

def df(i):
  f = open(TERM_INFO, 'r')
  li = list(f)
  return int(li[i].split('\t')[3])

def BM25(query, d, k1=1.2, k2=500, b=0.75):
  term_dic = TERMID_DICT
  score = 0
  if d in DOC_ID_TF.keys():
    try:
      K = k1 * ((1-b) + (b * (DOC_LEN[d] / AVG_LEN)))
      for q in query.split(' '):

        #handling the last word with '\n' case:
        # q = 1 word in query
        # refined is word without \n
        # i is the term id
        refined = q.split('\n')[0]
        if refined != '':
          i = term_dic[refined]
          i = int(i)
          #now this is i belongs to query;
          #we will work on these i's
          try:
            part1 = math.log10((TOTAL_DOC_IDS + 0.5)/(DF_DICT[i] + 0.5))
            tf_in_d =  DOC_ID_TF[d][i] if i in DOC_ID_TF[d].keys() else 0
            part2 = ((1+k1) * tf_in_d)/(K + tf_in_d)
            part3 = (1 + k2) / k2
          except Exception as e:
            pass

          score += (part1 * part2 * part3)
    except Exception as e:
      pass
  return score

def JM(query, d, param = 0.6):
  term_dic = TERMID_DICT
  prob = 1
  for q in query.split(' '):

    #handling the last word with '\n' case:
    # q = 1 word in query
    # refined is word without \n
    # i is the term id
    refined = q.split('\n')[0]
    if refined != '':
      i = term_dic[refined]
      i = int(i)
      #now this is i belongs to query;
      #we will work on these i's
      tf_in_d =  DOC_ID_TF[d][i] if i in DOC_ID_TF[d].keys() else 0
      
      part1 = param * (tf_in_d / DOC_LEN[d])
      
      sum_tf = 0
      for dd in range(TOTAL_DOC_IDS):
        if dd in DOC_ID_TF.keys():
          tf_in_dd =  DOC_ID_TF[dd][i] if i in DOC_ID_TF[dd].keys() else 0
          sum_tf += tf_in_dd
      try:
        part2 = (1 - param) * (sum_tf / sumLenD)
        prob *= (part1 + part2)
      except Exception as e:
        pass
  return prob

def query_tf_idf(tf,word):
    dfi=int(DF_DICT[word])
    return 0 if tf==0 else (math.log(tf) + 1)*math.log(TOTAL_DOC_IDS/dfi)
    
def find_cosine(q, d):
    return np.sum(np.multiply(q, d))/(np.linalg.norm(q)*np.linalg.norm(d))

def create_doc_id_tf_dict():
    #from DOC_INDEX, we find the doc_id then we get the frequency of each term in that doc into some data structure. form there we get the term frequenmcy
    #for each term in the doc.
    #we find the term frequency for each term in doc and then also create the query vector accoridngly
    with open(DOC_INDEX, 'r') as f:
        l = f.readlines()
        for line in l:
            line = line.split('\t')
            docid= int(line[0])
            termid= int(line[1])
            tf = int(line[2])
            if docid not in DOC_ID_TF.keys():
                DOC_ID_TF[docid] = {}
                DOC_ID_TF[docid][termid] = tf
            else:
                DOC_ID_TF[docid][termid] = tf

def TF_IDF(query, doc_id):
    '''
    query: array of str to search
    get_tfidf_from_corpus: function(word, doc_id)
    get_tfidf: function(tf)
    '''
    termids = TERMID_DICT

    query=query.split()
    for i in range(len(query)):
      query[i] = int(termids[query[i]])
    query=dict(Counter(query))
    

    
    union_of_terms = copy.deepcopy(query)
    union_of_terms.update(DOC_ID_TF[doc_id])
    d_vec=np.zeros(( len(union_of_terms.keys()) ,1 ))
    q_vec=np.zeros(( len(union_of_terms.keys()) ,1 ))
    
    for index, term in enumerate(union_of_terms):
        '''
        q is a term in the query
        doc_id is the id of the document for which we need to find the tfidf of the term
        query[q] is the frequency of the term in the query
        '''
        try:
          # print('ternm', term, termids)
          d_vec[index]=1+math.log10(DOC_ID_TF[doc_id][term]) if term in DOC_ID_TF[doc_id].keys() and DOC_ID_TF[doc_id][term] >0 else 0# get the tfidf for the doc
          
          tf_query = query[term] if term in query.keys() else 0
          
          q_vec[index]=query_tf_idf(tf_query,term)
        except Exception as e:
          pass
    
    return find_cosine(d_vec, q_vec)

class Result:
    def __init__(self, query_id, doc_id, score, rank = -1 ):
        self.query_id = query_id
        self.doc_id = doc_id
        self.rank = rank
        self.score = score
    
    def print(self):
        docname = DOC_NAMES[self.doc_id].split('\t')[1].split('\n')[0]        
        print(f'{self.query_id} 0 {docname} {self.rank} {self.score} run1')


def rank_results(function):
    index = 0 
    for query in list_of_queries:
      query_id = QUERY_IDS_LIST[index]
      unranked_results = []
      
      for doc_id in range(TOTAL_DOC_IDS):
        if doc_id in DOC_ID_TF.keys():
          temp_result = Result(query_id, doc_id, function(query, doc_id))
          unranked_results.append(temp_result)
      index = index + 1
        
      ranked_results = sorted(unranked_results, key= lambda x: x.score, reverse=True)
      r_index = 1
      for r in ranked_results:
          r.rank = r_index
          r_index += 1
          r.print()
    

if __name__ == '__main__':
  try:
    parser = argparse.ArgumentParser(description="Take a function name")

    parser.add_argument('--score', type=str,  action='store', required=True)
    args = parser.parse_args()
    create_doc_id_tf_dict()
    if os.path.exists(DOC_LEN_PATH):
      with open(DOC_LEN_PATH, 'rb') as f:
        DOC_LEN = pickle.load(f)
    else:   
      out_file = open(DOC_LEN_PATH, 'wb+')
      NUM_THREADS = 8
      THREADS = []
      # global DOC_LEN
      docid_list = list(DOC_ID_TF.keys())
      for x in range(0, len(docid_list), 8):
        threads = 0
        THREADS=[]
        while threads < NUM_THREADS and (x + threads < len(docid_list)):
          THREADS.append(threading.Thread(target=findLen, args=(docid_list[x+threads],)))
          THREADS[threads].start()
          threads+=1


        while 0< threads:
          THREADS[threads-1].join()
          threads-=1
      pickle.dump(DOC_LEN, out_file)
      out_file.close()
  
    TERMID_DICT = termid_dict()

    if args.score == 'TF-IDF':
      rank_results(TF_IDF)
    elif args.score == 'BM25':
      rank_results(BM25)
    elif args.score == 'JM':
      rank_results(JM)
      
  except Exception as e:
    logging.error(e, exc_info=True)
 
