def read_rawtext_as_words(fp:str, tk_sep=u'\u3000'):
    """Read a corpus from raw text

    Parameters
    ----------
    fp : str
        File path
    tk_sep : str, optional
        Token separator, by default u'\u3000'

    Yields
    -------
    str
        A word
    
    Details
    -------
    The structure of the (default) corpus file:
        <tk>\u3000<tk>\u3000<tk>\u3000<tk>\u3000...<tk>\n
        <tk>\u3000<tk>\u3000<tk>\u3000<tk>\u3000...<tk>\n
        ...
        <tk>\u3000<tk>\u3000<tk>\u3000<tk>\u3000...<tk>\n
    """
    with open(fp, encoding="utf-8") as f:
        for l in f:
            for w in l.strip().split(tk_sep):
                w = w.strip()
                if w == '': continue
                yield w