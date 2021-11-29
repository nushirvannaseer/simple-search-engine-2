from os import write
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

OUTPUT_FILE = 'data/queries.txt'

try:
    STOP_WORDS = [ word.strip() for word in open('data/stoplist.txt', 'r').readlines()]
except Exception as e:
    print(e)
    exit()

def remove_stop_words(query_words):
    return [word for word in query_words if word not in STOP_WORDS]
    
def apply_stemming(word_list):
    ps = PorterStemmer()
    stemmed_words = []
    for w in word_list:
        stemmed_words.append(ps.stem(w))
    return stemmed_words

def read_query_file(filename):
    try:
        f = open(filename, 'r')
        soup = BeautifulSoup(f, 'lxml')
        queries = soup.find_all('query')
        query_ids = soup.find_all('topic')
        for topic in query_ids:
            write_query_id_to_file(topic.get('number'))
        for query in queries:
            process_query(query.text)
        f.close()
    except Exception as e:
        print(e)
    
def write_query_id_to_file(id):
    with open('data/query_ids.txt', 'a+') as f:
        f.write(str(id)+'\n')

def process_query(query):
    #convert all to lower case
    query = query.lower()
    #split on whitespace
    query = query.split()
    #apply stop-wording
    query = remove_stop_words(query)
    #apply stemming
    query = apply_stemming(query)
    #write to file
    write_to_file(query)
    

def write_to_file(query):
    try:
        f = open(OUTPUT_FILE, 'a+')
        f.write(" ".join(query) + " \n")
        f.close()
    except Exception as e:
        print(e)
    
    

if __name__ == '__main__':
    read_query_file('data/topics.xml')
    