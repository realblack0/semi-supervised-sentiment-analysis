def read_news(filename):
    """파일 내용을 읽어온다."""
    with open(f"data/Contents/{filename}.txt", encoding="utf-8") as fp:
        doc = fp.read()
    return doc

def make_contents_column(newslist):
    """
    '파일'열의 파일명을 읽어서 'contents'열을 만든다.
    파일 경로는 'data/Contents/파일명.txt'이다.
    
    Arguments
    ---------
        doc(str) : 문서
    
    Return
    ------
        doc(str) : 정제된 문서
    """
    newslist["contents"] = newslist["파일명"].map(read_news)
    return newslist