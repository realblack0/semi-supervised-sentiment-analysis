import re
import matplotlib.pyplot as plt
from nltk.text import Text
from collections import defaultdict
from wordcloud import WordCloud
from math import log
from functools import reduce
from tqdm import tqdm

class PMI:
    def __init__(self, normalize, tokenize, cleansing, pos_tag):
        self.normalize = normalize
        self.tokenize = tokenize
        self.cleansing = cleansing
        self.pos_tag = pos_tag
        self._newslist = None
        self.exclude = set()
        
    def analysis(self, newslist):
        contents_dict = {}
        contents_dict["total"] = newslist.contents
        contents_dict["positive"] = newslist[newslist.Trend=="P"].contents
        contents_dict["negative"] = newslist[newslist.Trend=="N"].contents
        
        voca_dict = {"total":[]}
        for bag_of_words in newslist["contents"]:
            voca_dict["total"].extend(bag_of_words) 
    
        voca_dict["positive"] = reduce(lambda x, y: x+y, contents_dict["positive"], [])
        voca_dict["negative"] = reduce(lambda x, y: x+y, contents_dict["negative"], [])
        
        text_dict = {k:Text(v).vocab() for k,v in voca_dict.items()}
        
        self._contents_dict = contents_dict
        self._voca_dict = voca_dict
        self._text_dict = text_dict
        
    def process(self, newslist):
        """fit for data"""
        x = newslist.copy()
        x = self.normalize(x)
        x = self.tokenize(x)
        x = self.pos_tag(x)
        x = self.cleansing(x)
        self._newslist = x
        
        print("Start Analysis")
        self.analysis(x)
        
        print("All processes are done.")
        return self
        
    def post_process(self, exclude):
        self.exclude |= set(exclude)
        for ex in exclude:
            self._text_dict["total"].pop(ex, None)
            self._text_dict["positive"].pop(ex, None)
            self._text_dict["negative"].pop(ex, None)
            
    def info(self):
        for key, text in self._text_dict.items():
            print(key, "\n")
            print("total words:", text.N())
            print("unique words:", text.B())
            print()    
        
    def most_common(self, n=100):
        print(f"상위 {n}개 단어")
        print("순위      total     pos     neg")
        print("-"*40)
        for i, (total, pos, neg) in enumerate(zip(*[_.most_common(n) for _ in self._text_dict.values()])):
            print(f"{i}번째: {total}    {pos}    {neg}")
        
    @staticmethod
    def eliminate_pos(x):
        token, freq = x
        if token.count("/") > 1:
            return re.sub("/[A-Z]+", "", token), freq
        token, _ = token.split("/")
        return token, freq
    
    def plot_wordcloud(self, text, *, title=None, width=1000, height=600, figsize=(16,10), without=False):
        wordcloud = WordCloud(background_color="white", width=width, height=height)

        token_freq = dict(text.most_common())
        if without:
            token_freq = dict(map(self.eliminate_pos, token_freq.items())) # 버그: 동음이의어 충돌 위험

        wordcloud_text = wordcloud.generate_from_frequencies(token_freq)
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.axis(False)
        plt.imshow(wordcloud_text)
        plt.show()
    
    def plot(self, title=None, width=1000, height=600, figsize=(16,10), without=False):
        for title, t in self._text_dict.items():
            self.plot_wordcloud(t, title=title, width=width, height=height, figsize=figsize, without=without)
        
    def make_seeds(self, pre_seeds, verbose=1):
        """
        입력된 seed로 시작하는 token을 모두 찾는다.
        
        Arguments
        ---------
            pre_seeds(list[str]) : seed가 될 단어를 넣는다.
            verbose(int) : 0이면 단어와 빈도가 print되지 않는다.
        """
        seed_tokens = []
        for seed in pre_seeds:
            for token, freq in self._text_dict["total"].items():
                if token.startswith(seed):
                    seed_tokens.append(token)
                    if verbose:
                        print(token, freq)
        return seed_tokens
        
    def fit(self, pos, neg):
        """fit for PMI"""
        # TDM
        tdm = defaultdict(list)
        for index, content in self._newslist["contents"].items():
            for term in content:
                if term not in self.exclude:
                    tdm[term].append(index)
        
        # PMI
        N = len(self._newslist)
        pmi_lexicon = defaultdict(lambda : {"P":0, "N":0})


        for term, doc_frequency in tqdm(tdm.items()):
            for seed in pos:
                x = set(tdm[seed])
                y = set(doc_frequency)

                p_x = len(x) / N
                p_y = len(y) / N
                p_xy = len(x.intersection(y)) / N

                pmi_term_p = p_xy / (p_x*p_y)
                pmi_lexicon[term]["P"] += log(pmi_term_p) if pmi_term_p != 0 else 0 


            for seed in neg:
                x = set(tdm[seed])
                y = set(doc_frequency)

                p_x = len(x) / N
                p_y = len(y) / N
                p_xy = len(x.intersection(y)) / N

                pmi_term_n = p_xy / (p_x*p_y)
                pmi_lexicon[term]["N"] -= log(pmi_term_n) if pmi_term_n != 0 else 0
        
        self._tdm = tdm
        self._pmi_lexicon = pmi_lexicon
        
        return self
    
    def score_token_pmi(self, token):
        if token in self._pmi_lexicon:
            return self._pmi_lexicon[token]
        print(token, "is not in lexicon")
    
    def score_doc_pmi(self, content):
        """
        content의 pmi를 측정한다.
        
        Arguments
        ---------
            contents(list[token]) : token으로 된 list여야 한다.
        """
        so_pmi = 0
        for term in content:
            so_pmi += self._pmi_lexicon[term]["P"] 
            so_pmi += self._pmi_lexicon[term]["N"]
        return so_pmi
    
    def score_docID_pmi(self, documentID, *, verbose=1, without=False):
        content = self._newslist["contents"][documentID]
        pmi = self.score_doc_pmi(content)
        if verbose:
            print("Title", self._newslist["제목"][documentID])
            print("PMI:", pmi)
            print()
            if without:
                content = content.map(self.eliminate_pos)
            print(" ".join(content))
        return pmi
            
    def predict(self):
        newslist = self._newslist.copy()
        newslist["pmi"] = newslist["contents"].map(self.score_doc_pmi)
        return newslist