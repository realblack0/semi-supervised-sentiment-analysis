import nltk
from time import time
from datetime import timedelta
from nltk import word_tokenize
from functools import wraps

def time_log(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start = time()
        print("Start", fn.__qualname__.split(".")[0])
        result = fn(*args, **kwargs)
        print(f"End. (time: {str(timedelta(seconds=time() - start))})")
        return result
    return inner

class Normalize:
    def __init__(self, normalizers):
        """
        Arguments
        --------
            normalizers(list): 노말라이저 객체를 list로 묶음. 

        """
        self.normalizers = normalizers
        
    @time_log
    def __call__(self, newslist):
        for normalizer in self.normalizers:
            newslist = normalizer(newslist)
        return newslist
    
class Tokenize:
    def __init__(self, initial_tokenizer, post_tokenizers=None):
        """
        Arguments
        --------
            initial_tokeninzer(func): 토크나이저 함수. 함수는 str을 인자로 받고 token list를 반환해야함. 
            post_tokeninzer(list[func]): 토크나이저 함수로 된 리스트. 함수는 token list을 인자로 받고 token list를 반환해야함. 

        """
        if not initial_tokenizer:
            initial_tokenizer = word_tokenize
        self.initial_tokenizer = initial_tokenizer
        self.post_tokenizers = post_tokenizers

    @time_log
    def __call__(self, newslist):
        newslist["contents"] = newslist["contents"].map(self.initial_tokenizer)
        
        if self.post_tokenizers:
            for tokenizer in self.post_tokenizers:
                newslist["contents"] = newslist["contents"].map(tokenizer)
        return newslist
    
    
class Cleansing:
    def __init__(self, funcs):
        self.funcs = funcs
        
    @time_log
    def __call__(self, newslist):
        for func in self.funcs:
            newslist = func(newslist)
        return newslist
    

class PosTag:
    def __init__(self, use_pos=None):
        """
        단어 토큰에 품사를 달아서 동의어를 구분한다.

        Arguments
        ---------
            use_pos(set) : 사용할 태그목록을 지정함. 태그명은 `nltk.help.uppen_tagset`을 참조.
        """
        self.use_pos = use_pos
        
    def pos_tagger(self, tokens):
        tokens = nltk.pos_tag(tokens)
        if self.use_pos:
            tokens = [(_, pos) for _, pos in tokens if pos in self.use_pos]
        tokens = ["/".join([token, pos]) for token, pos in tokens]
        return tokens

    @time_log
    def __call__(self, newslist):
        """
        Input
        -----
            newslist의 contents열은 token list이어야 한다.
        
        Return
        -------
            tokens(list) : 'token/pos' 형태의 토큰으로 만들어줌.
        """
        newslist["contents"] = newslist["contents"].map(self.pos_tagger)
        # temp = []
        # for content in newslist["contents"]:
        #     temp.append(self.pos_tagger(content))
        # newslist["contents"] = pd.Series(temp)
        return newslist
