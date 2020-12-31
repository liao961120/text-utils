"""
Statistics for lexical items
"""
import math
import networkx as nx
from tqdm.auto import tqdm
from itertools import product
from collections import Counter
from typing import Union, Sequence
from .corp_readers import read_rawtext_as_words


class TF_IDF:
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

            .. code-block:: python

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


class ContextualDiversity:
    """Compute contextual diversity scores for words in a corpus
    """

    def __init__(self, fp:str, tk_sep:str=u'\u3000', WINDOW:int=4, SM:float=0.75):
        """Initialize corpus and compute PPMI co-occurrence scores

        Parameters
        ----------
        fp : str
            Path to plain text corpus file. Each line is a sentence,
            and tokens are separated by a separater specified in `tk_sep`
        tk_sep : str
            Token separator in the corpus raw text file
        WINDOW : int, optional
            Sliding window size used in PPMI computation, by default 4
        SM : float, optional
            Smoothing factor used in PPMI computation, by default 0.75
        
        References
        ----------
        Levy, O., Goldberg, Y., & Dagan, I. (2015). Improving Distributional Similarity with Lessons Learned from Word Embeddings. Transactions of the Association for Computational Linguistics, 3, 211–225.
        Jurafsky, D. (2015, July). Distributional (Vector) Semantics. https://web.stanford.edu/~jurafsky/li15/lec3.vector.pdf
        """

        # Construct corpus
        corp = list(read_rawtext_as_words(fp, tk_sep))
        corp_size = len(corp)

        # Count co-occurrences of all (target, context) pairs
        cooccur = {}
        target = {}
        context = {}
        word_freq = {}
        for i, tgt_w in enumerate(corp):
            # Record total freq
            if tgt_w not in word_freq: word_freq[tgt_w] = 0
            word_freq[tgt_w] += 1

            # Record cooccurences in context window
            s, e = max(0, i - WINDOW), min(corp_size, i + WINDOW + 1)
            l_cntx = corp[s:i]
            r_cntx = corp[(i+1):e] if i + 1 != corp_size else []
            
            for cntx_w in l_cntx + r_cntx:
                # Update cooccurrence count
                pair = (tgt_w, cntx_w)
                if pair not in cooccur: cooccur[pair] = 0
                cooccur[pair] += 1

                # Update target, context count
                if tgt_w not in target: target[tgt_w] = 0
                if cntx_w not in context: context[cntx_w] = 0
                target[tgt_w] += 1
                context[cntx_w] += 1

        self.tgt_w_freq = target
        self.cntx_w_freq = context
        self.cooccur_freq = cooccur
        self.w_freq = word_freq

        self.tgt_w_size = sum( v for k, v in self.tgt_w_freq.items() )
        self.cntx_w_size = sum(v**SM for k, v in self.cntx_w_freq.items())
        self.coocurr_size = sum( v for k, v in self.cooccur_freq.items() )
        self.SM = SM


    def diversity_scores(self, words:Sequence=None, return_graph=False, max_vocab_size:int=10000):
        """Quantify contextual diversity of each word in the corpus

        Parameters
        ----------
        words : Sequence, optional
            Words to compute scores, by default None
        return_graph : bool, optional
            Whether to return the co-occurence network of the words, 
            by default False
        max_vocab_size : int, optional
            The max size of vocabulary used to construct the word 
            co-occurrence network. Only the top-n (n = `max_vocab_size`) 
            frequent words in the corpus are used.

        Returns
        -------
        dict
            A dictionary recording contextual diversity scores of the
            words.
        
        Notes
        -----
        Implementation of the contexual diversity (polysemous) measure for 
        words in Hamilton et al. (2016). For the purpose of intuitive 
        interpretation, the original scores (local clustering coefficients) 
        are reversed by (1 - ori_score), such that a higher score indicates 
        higher contextual diversity of a word.

        References
        ----------
        Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic Word 
        Embeddings Reveal Statistical Laws of Semantic Change. In Proceedings 
        of the 54th Annual Meeting of the Association for Computational 
        Linguistics (pp. 1489–1501).
        """
        vocab = sorted(self.w_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
        vocab = [x[0] for x in vocab]

        # Construct co-occurrence network
        pbar = tqdm(total=len(vocab)**2)
        G = nx.DiGraph()
        for tgt_w, cntx_w in product(vocab, vocab):
            pbar.update()

            if tgt_w == cntx_w: continue
            val = self.pmi(tgt_w, cntx_w)
            if val > 0:
                G.add_edge(tgt_w, cntx_w)
                if return_graph:
                    G.edges[tgt_w, cntx_w]['ppmi'] = val
            else:
                G.add_node(tgt_w)
                G.add_node(cntx_w)
        pbar.close()

        local_clust_coef = nx.algorithms.cluster.clustering(G, nodes=words)
        diversity = { 
            k:(1-v) for k, v in local_clust_coef.items() 
        }
        if return_graph:
            return diversity, G
        return diversity


    def pmi(self, tgt_w: str, cntx_w: str):
        """Compute PMI score of a word pair

        Parameters
        ----------
        tgt_w : str
            Target word
        cntx_w : str
            Context word

        Returns
        -------
        float
            PMI value
        """
        Nwc = self.cooccur_freq.get((tgt_w, cntx_w), 0)
        if Nwc == 0: return -math.inf
        Nw = self.tgt_w_freq.get(tgt_w, 1)
        Nc = self.cntx_w_freq.get(cntx_w, 1)
        
        Pw = Nw / self.tgt_w_size
        Pc = Nc ** self.SM / self.cntx_w_size
        Pwc = Nwc / self.coocurr_size

        return math.log2( Pwc / (Pw * Pc) )
