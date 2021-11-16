import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    # pad the text with enough start tokens (n-1) 
    for i in range(1, n):
        text.insert(0, '<s>')
    # pad the text with a single end token 
    text.append('</s>')
    # for each "real" non-start token, create an n-gram tuple of the form (word, context), where word is a string and contet is the (n-1) tuple of preceding words/strings
    # print text[i:i+n] for i in range(len(txt)-(n-1))]
    for i in range(len(text)-(n-1)):
        context = tuple(text[i:i+n-1])
        word = text[i+n-1]
        ngram = (word, context)
        yield ngram
    pass


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    corpus_file = open(corpus_path, "r")
    corpus = corpus_file.read()
    paragraphs = corpus.split("\n\n")
    tokenized_sentences = []
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        # print(sentence)
        for sentence in sentences:
            words = word_tokenize(sentence)
            tokenized_sentences.append(words)
    return tokenized_sentences
    pass


# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    corpus = load_corpus(corpus_path)
    generateLM = NGramLM(n)
    for sentence in corpus:
        generateLM.update(sentence)
    return generateLM
    pass


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        ngrams = get_ngrams(n = self.n, text=text)
        for ngram in ngrams:
            self.vocabulary.add(ngram[0])
            if ngram[1] in self.context_counts:
                self.context_counts[ngram[1]] = self.context_counts[ngram[1]] + 1
            else:
                self.context_counts[ngram[1]] = 1
            if ngram in self.ngram_counts:
                self.ngram_counts[ngram] = self.ngram_counts[ngram]+1
            else:
                self.ngram_counts[ngram] = 1

        pass

    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        ngram = (word, context)
        # First, check if context is seen before
        if context in self.context_counts:
            # if yes, we can calculate pMLE by doing the division of the count of the ngram (word, context) by count(context)
            if ngram in self.ngram_counts:
                if (delta == 0):
                    return self.ngram_counts[ngram]/self.context_counts[context]
                else:
                    return (delta + self.ngram_counts[ngram])/(self.context_counts[context] + (delta * len(self.vocabulary)))
            else:
                if (delta == 0):
                    return 0
                else:
                    return (delta)/(self.context_counts[context] + (delta * len(self.vocabulary)))

        else:
            # if no, we return 1/|V|
            if (delta == 0):
                return 1/len(self.vocabulary)
            else:
                return delta/(delta * len(self.vocabulary))
        pass

    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        ngrams = get_ngrams(self.n, sent)
        log_prob_sums = 0
        for ngram in ngrams:
            prob = self.get_ngram_prob(ngram[0], ngram[1], delta)
            if(prob == 0):
                print ("zero prob for ngram: ", ngram)
                log_prob_sums = log_prob_sums -math.inf
            else:
                log_prob_sums = log_prob_sums + math.log(prob, 2)
        return log_prob_sums
        pass

    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]], delta=.0) -> float:
        # first flatten the list of lists of strings into a single list of strings
        tokens = 0
        for i in corpus:
            tokens = tokens + len(i)
        corpus_single_list = [j for sub in corpus for j in sub]
        sent_prob = self.get_sent_log_prob(corpus_single_list, delta)
        # sent_prob = sent_prob/len(corpus_single_list)
        sent_prob = sent_prob/tokens
        sent_prob = -1 * sent_prob
        return math.pow(2, sent_prob)
        pass

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        sorted_vocab = []
        for word in self.vocabulary:
            sorted_vocab.append(word)
        sorted_vocab.sort()
        r = random.random()
        cumulative_prob = 0
        for word in sorted_vocab:
            next_cumulative_prob = cumulative_prob + self.get_ngram_prob(word, context, delta)
            if (r >= cumulative_prob) and (r <= next_cumulative_prob):
                return word
            cumulative_prob = next_cumulative_prob
        pass

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        context = []
        for i in range(1, self.n):
            context.append("<s>")
        generated_words = 0
        sentence = []
        while(generated_words < max_length):
            # generate a new word
            # the context based on 
            new_word = self.generate_random_word(tuple(context[generated_words:generated_words+self.n]), delta)
            context.append(new_word)
            sentence.append(new_word)
            generated_words = generated_words + 1
            if(new_word == "</s>"):
                break
        sent_string = ""
        for word in sentence:
            sent_string = sent_string + word + " "
        return sent_string
        pass


def main(corpus_path: str, delta: float, seed: int):
    # trigram_lm = create_ngram_lm(3, corpus_path)
    # s1 = 'God has given it to me, let him who touches it beware!'
    # s2 = 'Where is the prince, my Dauphin?'

    # print(trigram_lm.get_sent_log_prob(word_tokenize(s1), 0.5))
    # print(trigram_lm.get_sent_log_prob(word_tokenize(s2), 0.5))
    # print(trigram_lm.get_perplexity([['God', 'has', 'given', 'to','it', 'me']], delta=.5))
    # print(trigram_lm.generate_random_word(('<s>', '<s>'), 0.4))
    # print(trigram_lm.generate_random_text(5, 0.4))

    print("***********")
    unigram_lm = create_ngram_lm(1, './warpeace.txt')
    for i in range(0, 5):
        print(unigram_lm.generate_random_text(10, 0.5))

    print("***********")
    trigram_lm = create_ngram_lm(3, './warpeace.txt')
    for i in range(0, 5):
        print(trigram_lm.generate_random_text(10, 0.5))

    print("***********")
    pentagram_lm = create_ngram_lm(5, './warpeace.txt')
    for i in range(0, 5):
        print(pentagram_lm.generate_random_text(10, 0.5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
