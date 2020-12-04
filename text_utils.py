import math
from collections import Counter
from typing import Union


###############################
# String manipulation
###############################
def strF2H(s):
    """全形轉半形

    Parameters
    ----------
    s : str
        含有全形字的字串

    Returns
    -------
    str
        全為半形字的字串

    Examples
    --------
    >>> strF2H('ａａａ')
    'aaa'
    """
    rstring = ""
    for uchar in s:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    return rstring


def strH2F(s):
    """半形轉全形

    Parameters
    ----------
    s : str
        含有半形字的字串

    Returns
    -------
    str
        全為全形字的字串

    Examples
    --------
    >>> strH2F('aaa')
    'ａａａ'
    """
    rstring = ""
    for uchar in s:
        u_code = ord(uchar)
        if u_code == 32:  # 全形空格直接轉換
            u_code = 12288
        elif 33 <= u_code <= 126:  # 全形字元（除空格）根據關係轉化
            u_code += 65248
        rstring += chr(u_code)
    return rstring


def str_replace(s, charset: str, replacement=''):
    """Replace or remove multiple characters from a string

    Parameters
    ----------
    s : str
        String to replace
    charset : str
        A string of characters to replace or remove from `s`
    replacement : str, optional
        Replacement string, by default '', which is equivalent
        to removing characters in `charset` from `s`

    Returns
    -------
    str
        The string with replacement inserted

    Examples
    --------
    >>> str_replace('abcde', 'ce', '_')
    'ab_d_'
    >>> str_replace('abcde', 'ce')
    'abd'
    """
    for char in charset:
        s = s.replace(char, replacement)
    return s


###############################
# Chinese character processing
###############################
def has_zh(x: str):
    """Check whether a string contains Chinese characters

    Parameters
    ----------
    x : str
        String to check

    Returns
    -------
    bool
        True if the input contains Chinese character, else False
    """
    for char in x:
        if (char > u'\u4e00' and char < u'\u9fff') or (char > u'\u3400' and char < u'\u4DBF'):
            return True
    return False


def all_zh(x: str):
    """Check whether a string is only comprised of Chinese characters

    Parameters
    ----------
    x : str
        String to check

    Returns
    -------
    bool
        True if the input string has only Chinese characters, else False
    """
    for char in x:
        if not ((char > u'\u4e00' and char < u'\u9fff') or (char > u'\u3400' and char < u'\u4DBF')):
            return False
    return True


###############################
# Text processing
###############################
class TF_IDF():
    """Compute tf-idf scores for terms in a list of documents
    """

    def __init__(self, docset: Union[dict, list]):
        """Initialize a TF_IDF obj

        Parameters
        ----------
        docset : dict or list
            A dict with key being document name
            and value being document content or a list of documents. 
            The content of each document is a list of tokens.

        Raises
        ------
        Exception
            `docset` should be a 2-level nested list, with the first level
            being the documents and the second level being the tokens in the 
            documents
        """
        self.N = len(docset)
        self.words = set()
        self.doc_ids = []
        self.doc_word_freq = {}  # 記錄資訊以供 self._tf() 計算
        self.doc_size = {}       # 記錄資訊以供 self._tf() 計算
        self.idf_cache = {}      # 記錄已經算過的 idf 分數

        if isinstance(docset, list):
            docset = {i: content for i, content in enumerate(docset)}
        for docname, content in docset.items():
            if not isinstance(content, list):
                raise Exception(
                    f'document content should be a list of tokens, not type: {type(content)}')
            self.doc_size[docname] = len(content)
            self.doc_word_freq[docname] = Counter(content)
            self.doc_ids.append(docname)
            for word in content:
                self.words.add(word)

    def _tf(self, word: str, doc_id):
        return self.doc_word_freq[doc_id][word] / self.doc_size[doc_id]

    def _idf(self, word: str):
        if word in self.idf_cache:
            return self.idf_cache[word]
        df = 0
        for _, doc in self.doc_word_freq.items():
            if word in doc:
                df += 1
        idf = math.log(self.N/df)
        self.idf_cache[word] = idf
        return idf

    def score(self, word: str, doc_id):
        """Compute tf-idf score of a term in a particular document

        Parameters
        ----------
        word : str
            The term to compute the score
        doc_id : int or str
            The id specifying the document in `docset` when initializing
            the TF_IDF obj

        Returns
        -------
        float
            The tf-idf score of the term

        Examples
        --------
        >>> docset = {
            'doc0': ['a', 'a', 'a', 'c'],
            'doc1': ['a', 'a', 'a', 'a', 'c'],
            'doc2': ['b', 'c', 'c', 'c']
        }
        >>> tfidf = TF_IDF(docset)
        >>> tfidf.score('a', 'doc1')
        0.32437208648653154
        >>> docset = [
            ['a', 'a', 'a', 'c'],
            ['a', 'a', 'a', 'a', 'c'],
            ['b', 'c', 'c', 'c']
        ]
        >>> tfidf = TF_IDF(docset)
        >>> tfidf.score('a', 1)
        0.32437208648653154
        """
        if word not in self.words:
            print(f"`{word}` not found in docset")
            return None
        return self._tf(word, doc_id) * self._idf(word)

    def score_all(self):
        """Compute tf-idf scores for all terms in the docset

        Returns
        -------
        dict
            A dict in the format of:
            {
                "<docid>": {
                    "<word>": <tf-idf-score>,
                    "<word>": <tf-idf-score>,
                    ...
                },
                "<docid>": {...},
                ...
            }

        Examples
        --------
        >>> docset = {
            'doc0': ['a', 'a', 'a', 'c'],
            'doc1': ['a', 'a', 'a', 'a', 'c'],
            'doc2': ['b', 'c', 'c', 'c']
        }
        >>> tfidf = TF_IDF(docset)
        >>> tfidf.score_all()
        {'doc0': {'a': 0.3040988310811233, 'c': 0.0}, 'doc1': {'a': 0.32437208648653154, 'c': 0.0}, 'doc2': {'b': 0.27465307216702745, 'c': 0.0}}
        """
        out = {}
        for docid in self.doc_ids:
            out[docid] = {}
            for word in self.doc_word_freq[docid]:
                out[docid][word] = self.score(word, docid)
        return out
