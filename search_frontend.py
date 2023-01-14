from flask import Flask, request, jsonify
import itertools
import operator
from nltk.corpus import stopwords
import numpy as np
from inverted_index_gcp import *
import math
from numpy.linalg import norm
from collections import defaultdict
from nltk.stem.porter import *


class MyFlaskApp(Flask):
    def __init__(self, import_name):
        super().__init__(import_name)
        # body index
        self.index = InvertedIndex.read_index('body_index', 'index')
        # helper index , contains a data about DL, avgDL,number of docs in the corpus
        self.data = InvertedIndex.read_index('body_index', 'data')
        self.body_DL = self.data.DL
        # title index
        self.title_index = InvertedIndex.read_index('title_index', 'title_new')
        # dictionary that hold docID as keys, title as value
        self.title_dict = self.title_index.title_dict
        self.title_DL = self.title_index.DL
        # anchor index
        self.anchor_index = InvertedIndex.read_index('anchor_index', 'anchor')
        # reading the pageRank file
        self.pr = pd.read_csv("part-00000-c56565c7-b8e5-4342-a340-9834e0e40a5a-c000.csv.gz",
                              names=['doc_id', 'rank'], header=None)
        with open('pageView.pkl', 'rb') as f:
            # pageView file
            self.pv = pickle.load(f)
        # title index with stemming, used in the search function
        self.stemmed_title = InvertedIndex.read_index('title_stemmed', 'title_stemming')

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def tokenize_title(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in stopwords_frozen]
    return list_of_tokens


# we checked if the query contains only one token. if that's the case, we use boolean ranking on the stemmed title_index.
# else, we use BM25 on the title index and on the body index, we give weights to each score, bodyweihgt is 0.7, titleweight is 0.3
# with k = 1.2 , b = 0.75.

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # # BEGIN SOLUTION
    k1 = 1.2
    b = 0.75
    w_title = 0.3
    w_body = 0.7
    tok = tokenize_title(query)
    w_t_q = Counter(tok)
    sim_t = defaultdict()
    t_pl = []
    stemmer = PorterStemmer()

    if len(tok) <= 1:
        tok = [stemmer.stem(t) for t in tok]
        for term in tok:
            if term in app.stemmed_title.df:
                t_pl += (app.stemmed_title.read_posting_list(term, 'title_stemmed'))
        doc_dict = {}
        for tup in t_pl:
            if tup[0] not in doc_dict:
                doc_dict[tup[0]] = 1
            else:
                doc_dict[tup[0]] += 1

        doc_dict = sorted([(doc_id, score) for doc_id, score in doc_dict.items()], key=lambda x: x[1], reverse=True)[
                   :40]
        for doc in doc_dict:
            res.append((doc[0], app.title_dict[doc[0]]))
    else:
        for term in tok:
            if term in app.title_index.df:
                idf_q_at = math.log(
                    1 + (app.data.N - app.title_index.df[term] + 0.5) / (app.title_index.df[term] + 0.5))
                t_pl = app.title_index.read_posting_list(term, 'title_index')
                for doc, tf in t_pl:
                    if (doc, query) not in sim_t:
                        sim_t[(doc, query)] = (idf_q_at * tf * (k1 + 1) * w_t_q[term]) / \
                                              (tf + k1 * (1 - b + b * app.title_index.DL[doc] / app.title_index.AVGDL))
                    else:
                        sim_t[(doc, query)] += (idf_q_at * tf * (k1 + 1) * w_t_q[term]) / \
                                               (tf + k1 * (1 - b + b * app.title_index.DL[doc] / app.title_index.AVGDL))
        sim_t = sorted([(d_q, score * w_title) for d_q, score in sim_t.items()], key=lambda x: x[1], reverse=True)

        q = tokenize(query)
        w_t_q = Counter(q)
        idf_q = {}
        sim = defaultdict()
        for term in q:
            if term in app.index.df:
                idf_q = math.log(1 + (app.data.N - app.index.df[term] + 0.5) / (app.index.df[term] + 0.5))
                pl = app.index.read_posting_list(term, 'body_index')
                for doc, tf in pl:
                    if (doc, query) not in sim:
                        sim[(doc, query)] = ((idf_q * tf * (k1 + 1) * w_t_q[term]) / \
                                             (tf + k1 * (1 - b + b * (app.data.DL[doc] / app.data.AVGDL)))) * w_body
                    else:
                        sim[(doc, query)] += ((idf_q * tf * (k1 + 1) * w_t_q[term]) / \
                                              (tf + k1 * (1 - b + b * (app.data.DL[doc] / app.data.AVGDL)))) * w_body

        for d_q, score in sim_t:
            if d_q not in sim:
                sim[d_q] = score
            else:
                sim[d_q] += score

        for d_q, score in sim.items():
            if d_q[0] in app.pv:
                page = app.pv[d_q[0]]
                sim[d_q] = (score * page) / (score + page)
        sim = sorted([(d_q, score) for d_q, score in sim.items()], key=lambda x: x[1], reverse=True)[:100]

        for x, score in sim:
            if x[0] in app.title_dict:
                title = app.title_dict[x[0]]
                tup = (x[0], title)
                res.append(tup)
    # END SOLUTION
    return jsonify(res)


# we use cosine similarity to return the most relative documents.
@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    q = tokenize(query)
    w_t_q = Counter(q)
    sim = defaultdict()
    for term in np.unique(q):
        if term in app.index.df:
            pl = app.index.read_posting_list(term, 'body_index')
            for d, tf in pl:
                if (d, query) not in sim:
                    sim[(d, query)] = (w_t_q[term] / len(q)) * \
                                      (math.log(app.data.N / app.index.df[term], 10) * tf / app.data.DL[d])
                else:
                    sim[(d, query)] += (w_t_q[term] * math.log(app.data.N / app.index.df[term], 10) / len(q)) * \
                                       (math.log(app.data.N / app.index.df[term], 10) * tf / app.data.DL[d])
    sum = 0
    for val in w_t_q.values():
        sum += math.pow(val / len(q), 2)
    q_j = math.sqrt(sum)
    for d_q, score in sim.items():
        sim[d_q] = score * (1 / q_j) * (1 / app.data.nf_dict[d_q[0]])
    sim = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:100]

    for x, score in sim:
        if x[0] in app.title_dict:
            title = app.title_dict[x[0]]
            tup = (x[0], title)
            res.append(tup)
    # END SOLUTION
    return jsonify(res)


# we use boolean ranking to return he documents which contains the most words from the query
@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    t_pl = []
    tok = tokenize_title(query)
    for term in tok:
        if term in app.title_index.df:
            t_pl += (app.title_index.read_posting_list(term, 'title_index'))
    doc_dict = {}
    for tup in t_pl:
        if tup[0] not in doc_dict:
            doc_dict[tup[0]] = 1
        else:
            doc_dict[tup[0]] += 1
    doc_dict = sorted([(doc_id, score) for doc_id, score in doc_dict.items()], key=lambda x: x[1], reverse=True)
    for doc in doc_dict:
        doc_title = (doc[0], app.title_dict[doc[0]])
        res.append(doc_title)

    # END SOLUTION
    return jsonify(res)


# same with search_title, only we do this on the anchor index.
@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    a_pl = []
    tok = tokenize_title(query)
    for term in tok:
        if term in app.anchor_index.df:
            a_pl += (app.anchor_index.read_posting_list(term, "anchor_index"))
    doc_dict = {}
    for tup in a_pl:
        if tup[0] not in doc_dict:
            doc_dict[tup[0]] = 1
        else:
            doc_dict[tup[0]] += 1
    doc_dict = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
    for doc in doc_dict:
        if doc[0] in app.title_dict:
            title = app.title_dict[doc[0]]
            tup = (doc[0], title)
            res.append(tup)

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [float(app.pr['doc_id'][i]) for i in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = [app.pv[i] for i in wiki_ids]
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
