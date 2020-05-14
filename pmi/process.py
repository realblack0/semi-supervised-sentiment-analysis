import re
import nltk
from string import punctuation

class ExtractArticle:
    def extract_article(self, doc):
        """
        뉴스기사 txt파일에서 본문만 가져온다.

        Arguments
        ---------
            doc(str) : 문서

        Return
        ------
            doc(str) : 정제된 문서
        """
        return "".join(doc.splitlines()[6:])
        
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.extract_article)
        return newslist
    

class ConcatenateHead:
    def __call__(self, newslist):
        newslist["contents"] = newslist.fillna("").apply(lambda x: x["제목"] + " " + x["contents"], axis=1)
        return newslist

class LowerAlphabet:
    def lower_alphabet(self, doc):
        """
        뉴스기사를 소문자로 정규화한다.

        Arguments
        ---------
            doc(str) : 문서

        Return
        ------
            doc(str) : 정제된 문서
        """
        return doc.lower()
        
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.lower_alphabet)
        return newslist
        
        
class NormalizeSynonym:
    def __init__(self, thesaurus):
        self.thesaurus = thesaurus
        
    def normalize_synonym(self, doc):
        """동의어를 하나의 표현으로 통일한다.
        Arguments
        ---------
            doc(str) : 문서
            thesaurus(dict) : {대표 단어[str]: 동의어들[list]}

        Return
        ------
            doc(str) : 정제된 문서
        """
        for word, synonyms in self.thesaurus.items():
            doc = re.sub("|".join(synonyms), word, doc)
        return doc
        
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.normalize_synonym)
        return newslist
    
    
class EliminateSpecialCharacter:
    def __init__(self, save_punct=True, save_num=True, additional=None):
        """
        기본 pattern : A-Za-z_ \.
        save_punct : include string.punctuation
        save_num : 0-9
        additional : regex pattern
        """
        self.save_punct = save_punct
        self.save_num = save_num
        self.additional = additional
        
    def eliminate_special_character(self, doc):
        """특수문자를 제거하고 영어, 숫자, 공백, 문법 표현(특수문자-옵션)만 남긴다.
        영어 이외의 한자, 한글, 일본어, 아랍어 등은 지워진다."""
        pattern = "A-Za-z_ \."
        if self.save_punct:
            pattern += punctuation
        if self.save_num:
            pattern += "0-9"
        if self.additional:
            pattern += self.additional
        pattern = "[^" + pattern + "]"

        return re.sub(pattern, " ", doc)
        
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.eliminate_special_character)
        return newslist        
    
    
class PunctSpace:
    def punct_space(self, doc):
        """ '.' 뒤에 공백을 하나 붙인다."""
        return re.sub("\.", ". ", doc)
    
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.punct_space)
        return newslist
    
    
class Stopwords:
    def __init__(self, stopwords=None, *, additionals=None):
        """
        Arguments
        ---------
            stopwords(list[str]) : 불용어 리스트. 기본값은 nltk 영어 불용어.
            additionals(list[str]) : 기본값에 불용어를 추가할 때 사용.

        """
        if not stopwords:
            stopwords = nltk.corpus.stopwords.words("english")
        self.stopwords = stopwords
        if additionals:
                self.stopwords.extend(additionals)
        
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(lambda x:[_ for _ in x if _ not in self.stopwords])
        return newslist
    
class RegularLength:
    def __init__(self, minlen=0, maxlen=0):
        # assert minlen <= maxlen
        self.minlen = minlen
        self.maxlen = maxlen
        
    def __call__(self, newslist):
        if not self.minlen and not self.maxlen:
            return newslist
        
        elif self.minlen and not self.maxlen:
            newslist["contents"] = newslist["contents"].map(lambda x:[_ for _ in x if self.minlen <= len(_)])
            return newslist
        
        elif not self.minlen and self.maxlen:
            newslist["contents"] = newslist["contents"].map(lambda x:[_ for _ in x if len(_) <= self.maxlen])
            return newslist
        
        else:
            newslist["contents"] = newslist["contents"].map(lambda x:[_ for _ in x if self.minlen <= len(_) <= self.maxlen])
            return newslist