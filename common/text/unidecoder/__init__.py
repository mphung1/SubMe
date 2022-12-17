import io
import warnings

from .homoglyphs import homoglyphs
from .replacements import replacements

warnings.simplefilter('always', UserWarning)


_replacements = {uni: asc for uni, asc in replacements}
_homoglyphs = {g: asc for asc, glyphs in homoglyphs.items() for g in glyphs}


def unidecoder(s, homoglyphs=False):
    """Transliterate unicode

    Args:
        s (str): unicode string
        homoglyphs (bool): prioritize translating to homoglyphs
    """
    warned = False  # Once per utterance
    ret = ''
    for u in s:
        if ord(u) < 127:
            a = u
        elif homoglyphs:
            a = _homoglyphs.get(u, _replacements.get(u, None))
        else:
            a = _replacements.get(u, _homoglyphs.get(u, None))

        if a is None:
            if not warned:
                warnings.warn(f'Unexpected character {u}: '
                              'please revise your text cleaning rules.',
                              stacklevel=10**6)
                warned = True
        else:
            ret += a

    return ret
