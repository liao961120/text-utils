"""
Utility functions for string processing
"""
import re
import math
import networkx as nx
from tqdm.auto import tqdm


###############################
# String manipulation
###############################
def strF2H(x):
    """全形轉半形

    Parameters
    ----------
    x : str
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


def strH2F(x):
    """半形轉全形

    Parameters
    ----------
    x : str
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


def str_replace(x, charset: str, replacement=''):
    """Replace or remove multiple characters from a string

    Parameters
    ----------
    x : str
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
    if x == '': return False
    for char in x:
        if not ((char > u'\u4e00' and char < u'\u9fff') or (char > u'\u3400' and char < u'\u4DBF')):
            return False
    return True
