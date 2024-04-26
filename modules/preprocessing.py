from dataclasses import dataclass
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    content = open(filename, 'r').read().replace('&', '&amp;')
    tree = ET.fromstring(content)
    if tree.tag != 'sentences':
        raise ValueError('wrong type of xml file')

    sents_pair = []
    label_pair = []
    
    for sentence in tree:
        eng = sentence.find('english').text; zch = sentence.find('czech').text
        sen_pair = SentencePair(eng.split(), zch.split())
        sents_pair.append(sen_pair)

        sures = sentence.find('sure').text
        possibles = sentence.find('possible').text
        sure_pair = [tuple(map(lambda x: int(x), p.split('-'))) for p in sures.split()] if sures is not None else []
        poss_pair = [tuple(map(lambda x: int(x), p.split('-'))) for p in possibles.split()] if possibles is not None else []
        labl_pair = LabeledAlignment(sure_pair, poss_pair)
        label_pair.append(labl_pair)

    return sents_pair, label_pair


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    def _build_dict(old_map, freq_cutoff):
        mapping = np.array(list(old_map.items()))
        keys = mapping[:, 0]
        freqs = mapping[:, 1].astype(np.int32)
        if freq_cutoff is not None and freq_cutoff < len(freqs):
            tops = np.argpartition(freqs * -1, freq_cutoff)[:freq_cutoff]

        return {key: i for i, key in enumerate(keys) if freq_cutoff is None or freq_cutoff >= len(freqs) or i in tops}
    
    eng_tokens = defaultdict(lambda: 0)
    zch_tokens = defaultdict(lambda: 0)

    for pair in sentence_pairs:
        for eng_word in pair.source:
            eng_tokens[eng_word] += 1
        for zch_word in pair.target:
            zch_tokens[zch_word] += 1

    eng_tokens = _build_dict(eng_tokens, freq_cutoff)
    zch_tokens = _build_dict(zch_tokens, freq_cutoff)

    return eng_tokens, zch_tokens


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized = []
    eng_vocab_words = source_dict.keys()
    zch_vocab_words = target_dict.keys()

    for pair in sentence_pairs:
        tokens_source = [source_dict[eng] for eng in pair.source if eng in eng_vocab_words]
        tokens_target = [target_dict[zch] for zch in pair.target if zch in zch_vocab_words]
        
        if not len(tokens_source) or not len(tokens_target):
            continue
        
        tokenized.append(TokenizedSentencePair(tokens_source, tokens_target))
    
    return tokenized
