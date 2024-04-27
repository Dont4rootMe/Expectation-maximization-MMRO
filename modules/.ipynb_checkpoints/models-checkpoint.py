from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np
import scipy

from modules.preprocessing import TokenizedSentencePair


class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        raise NotImplementedError()

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        raise NotImplementedError()


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner_SGD(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters, m_steps=5, learning_rate=1e-2):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters
        self.m_steps = m_steps
        self.learning_rate = learning_rate

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        # create matrix with probabilities
        #      [count-sentences x sentence-source-words x sentence-target-words]
        connections = []
        
        # iterate over each pair of sentences
        for i, corp in enumerate(parallel_corpus):
            # get tokens of each language in pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # for each connection get probability
            #   acquire probability norm for each target word alignment
            #   and then get each aposteory probability
            src_trg_indx = np.ix_(src, trg)
                  
            connections.append(self.translation_probs[src_trg_indx])
            connections[i] /= np.sum(self.translation_probs[src_trg_indx], axis=0)

        return connections


    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
            
        Tips: 
            1) Compute mathematical expectation with a constant
            2) It is preferred to write this computation with 1 cycle only
        """
        # setting the sum over each sentece
        L_sum = 0

        # loop over each sentece
        for i, corp in enumerate(parallel_corpus):
            # get tokens of each language in pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # getting posterior probas for each token-alignments in sentence
            q_i = posteriors[i]

            # for each connection get log-probability
            src_trg_indx = np.ix_(src, trg)
            log_probas = np.log(self.translation_probs[src_trg_indx] + 1e-20)
            log_inv_n = -np.log(len(src))

            # multiply posterior and log probas and sum over L_sum
            L_sum += np.sum(q_i * log_probas)
            # add prob_norm const
            L_sum += len(trg) * log_inv_n
            # add Fisher-Informative index
            L_sum -= np.sum(q_i * np.log(q_i + 1e-20))

        return L_sum


    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        # gradient initialization
        grad = np.zeros_like(self.translation_probs)
        # zero-initialization of logits
        logits = scipy.special.logit(self.translation_probs)
        
        for i in range(self.m_steps):
            # computing derivative of ln(sigmoid(theta))
            logits_exp = np.exp(-logits)
            d_sigmoid = logits_exp / (1 + logits_exp)

            # iteration over each sentence
            for i, corp in enumerate(parallel_corpus):
                # getting tokens of each sentence pair
                src = corp.source_tokens
                trg = corp.target_tokens

                # acquire indexes and add sent-grad input to whole gradient
                # src_trg_indx = np.ix_(src, trg)
                sents_grad = posteriors[i] * d_sigmoid[src_trg_indx]
                grad[src_trg_indx] += sents_grad

            # make gradient step
            logits += self.learning_rate * grad

            # norm probabilities
            pseudo_probs = scipy.special.expit(logits)
            norms = pseudo_probs.sum(axis=-1)
            logits = scipy.special.logit(pseudo_probs / norms[:, None])

        # transform logits to probas
        self.translation_probs = scipy.special.expit(logits)
        return self._compute_elbo(parallel_corpus, posteriors)


    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    def align(self, sentences):
        # create list of alignments over sentences
        result = []
        for pair in sentences:
            # getting tokens of each sentence pair
            src = pair.source_tokens
            trg = pair.target_tokens
            
            # make map of indexes for tokens
            src_trg_indx = np.ix_(src, trg)
            probas = self.translation_probs[src_trg_indx]

            # define best alignment as max-prob align
            ind_alignments = probas.argmax(axis=0)
            # add 1 to each coordinate as it is formatted in initial corpus
            alignment = [(j + 1, i + 1) for i, j in enumerate(ind_alignments)]

            result.append(alignment)

        return result


class WordAligner(WordAligner_SGD):
    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        # pseudo-probas initialization
        self.translation_probs.fill(0.0)

        # iteration over each sentence
        for i, corp in enumerate(parallel_corpus):
            # getting tokens of each sentence pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # acquire indexes and add sent-grad input to whole psp
            src_trg_indx = np.ix_(src, trg)
            np.add.at(self.translation_probs, src_trg_indx, posteriors[i])

        self.translation_probs /= np.sum(self.translation_probs, axis=1)[:, None]
        
        return self._compute_elbo(parallel_corpus, posteriors)


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        if (src_length, tgt_length) not in self.alignment_probs.keys():
            self.alignment_probs[(src_length, tgt_length)] = np.full((src_length, tgt_length), 1 / src_length, dtype=np.float32)

        return self.alignment_probs[(src_length, tgt_length)]

    def _e_step(self, parallel_corpus):
        # create matrix with probabilities
        #      [count-sentences x sentence-source-words x sentence-target-words]
        connections = []
        
        # iterate over each pair of sentences
        for i, corp in enumerate(parallel_corpus):
            # get tokens of each language in pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # for each connection get probability of aposterior token alignment in this position
            #   acquire probability norm for each target word alignment
            #   and then get each aposteory probability
            src_trg_indx = np.ix_(src, trg)

            # get prior probs of conections
            connections.append(self.translation_probs[src_trg_indx])
            # get and mult prior probs of positions
            connections[i] *= self._get_probs_for_lengths(len(src), len(trg))
            connections[i] /= np.sum(connections[i], axis=0)

        return connections

    def _compute_elbo(self, parallel_corpus, posteriors):
        # setting the sum over each sentece
        L_sum = 0

        # loop over each sentece
        for i, corp in enumerate(parallel_corpus):
            # get tokens of each language in pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # getting posterior probas for each token-alignments in sentence
            q_i = posteriors[i]

            # for each connection get log-probability
            src_trg_indx = np.ix_(src, trg)
            log_probas = np.log(self.translation_probs[src_trg_indx] + 1e-20) + np.log(self._get_probs_for_lengths(len(src), len(trg)) + 1e-20)

            # multiply posterior and log probas and sum over L_sum
            L_sum += np.sum(q_i * log_probas)
            # add Fisher-Informative index
            L_sum -= np.sum(q_i * np.log(q_i + 1e-20))

        return L_sum

    def _m_step(self, parallel_corpus, posteriors):
        # pseudo-probas initialization
        self.translation_probs.fill(0.0)
        for value in self.alignment_probs.values():
            value.fill(0.0)

        # iteration over each sentence
        for i, corp in enumerate(parallel_corpus):
            # getting tokens of each sentence pair
            src = corp.source_tokens
            trg = corp.target_tokens

            # acquire indexes and add sent-grad input to whole psp
            src_trg_indx = np.ix_(src, trg)
            np.add.at(self.translation_probs, src_trg_indx, posteriors[i])
            
            if (len(src), len(trg)) not in self.alignment_probs.keys():
                self.alignment_probs[(len(src), len(trg))]  = posteriors[i]
            else:
                self.alignment_probs[(len(src), len(trg))] += posteriors[i]

        self.translation_probs /= np.sum(self.translation_probs, axis=1)[:, None]
        for value in self.alignment_probs.values():
            value /= np.sum(value, axis=0)[None, :]
        
        return self._compute_elbo(parallel_corpus, posteriors)
