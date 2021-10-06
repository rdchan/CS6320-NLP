import sys

import nltk
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from itertools import groupby
# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)

# helper function for get_ngram_features
# words is a list of strings
# i is an int
# Returns the list at i if it exists, and the starting/ending symbols if it's out of bounds
def check_bounds(words, i):
    if (i < 0):
        return '<s>'
    elif (i > len(words)-1):
        return '</s>'
    else:
        return words[i]


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings

def get_ngram_features(words, i):
    features = []
    features.append('prevbigram-'+check_bounds(words, i-1))
    features.append('nextbigram-'+check_bounds(words, i+1))
    features.append('prevskip-'+check_bounds(words, i-2))
    features.append('nextskip-'+check_bounds(words, i+2))
    features.append('prevtrigram-'+check_bounds(words, i-1)+'-'+check_bounds(words, i-2))
    features.append('nexttrigram-'+check_bounds(words, i+1)+'-'+check_bounds(words, i+2))
    features.append('centertrigram-'+check_bounds(words, i-1)+'-'+check_bounds(words, i+1))
    return features
    pass


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    features = []
    features.append('word-'+word)
    if (word[0].isupper()):
        features.append('capital')
    if (word.isupper()):
        features.append('allcaps')
    wordshape_list = ['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in word]
    features.append('wordshape-'+''.join(wordshape_list))
    features.append('short-wordshape-'+''.join([i[0] for i in groupby(wordshape_list)]))
    if(any([c.isdigit() for c in word])):
        features.append('number')
    if(any([c == '-' for c in word])):
        features.append('hyphen')
    for j in range(1, 5):
        if(j < len(word)+1 ):
            features.append('prefix'+str(j)+'-'+word[0:j])
    for j in range(1, 5):
        if(len(word)-j+1 > 0):
            features.append('suffix'+str(j)+'-'+word[len(word)-j:])
    return features
    pass


# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    full_features = []
    full_features.extend(get_ngram_features(words, i))
    full_features.extend(get_word_features(words[i]))
    full_features.append('tagbigram-'+prevtag)
    full_features = [f if 'wordshape' in f else f.lower() for f in full_features]
    return full_features
    pass


# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    rare_features = set()
    common_features = set()
    feature_counts = {}
    for sentence in corpus_features:
        for word_feature_list in sentence:
            for feature in word_feature_list:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                    if (feature_counts[feature] == threshold):
                        common_features.add(feature)
                else:
                    feature_counts[feature] = 1

    corpus_features_rare_removed = [[[feature for feature in word_list if feature in common_features] for word_list in sentence] for sentence in corpus_features]
    return (corpus_features_rare_removed, common_features)
    pass


# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    position_idx = 0
    feature_dict = {}
    for feature in common_features:
        if not feature in feature_dict:
            feature_dict[feature] = position_idx
            position_idx = position_idx + 1

    position_idx = 0
    tag_dict = {}
    for sentence in corpus_tags:
        for word_tag in sentence:
            if not word_tag in tag_dict:
                tag_dict[word_tag] = position_idx
                position_idx = position_idx + 1
    return (feature_dict, tag_dict)
    pass

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    Y = []
    for sentence in corpus_tags:
        for tag in sentence:
            Y.append(tag_dict[tag])
    Y = numpy.array(Y)
    print("build y length", len(Y))
    return Y
    pass

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    rows = []
    cols = []
    for idx, sentence in enumerate(corpus_features):
        for word in sentence:
            for feature in word:
                rows.append(idx)
                cols.append(feature_dict[feature])
    values = [1] * len(rows)
    print("build x rows", len(rows))
    rows =numpy.array(rows)
    cols =numpy.array(cols)
    values =numpy.array(values)
    return csr_matrix((values, (rows, cols)))

    pass


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    corpus_sentences, corpus_tags = load_training_corpus(proportion)
    # corpus_features is a list of lists of feature lists
    # it is a list of sentences
    # the sentences are a list of lists, each corresponding to a word
    # the sublists are the word features
    corpus_features = []
    for sentence in corpus_sentences:
        sentence_features = []
        prevtag = '<s>'
        for idx, word in enumerate(sentence):
            word_features = get_features(sentence, idx, prevtag)
            prevtag = word
            sentence_features.append(word_features)
        corpus_features.append(sentence_features)

    corpus_features, common_features =  remove_rare_features(corpus_features)
    feature_dict, tag_dict = get_feature_and_label_dictionaries(common_features, corpus_tags)
    Y = build_Y(corpus_tags, tag_dict)
    X = build_X(corpus_features, feature_dict)
    # print('tain deb')
    # # print(type(X))
    print(X.shape)
    lrmodel = LogisticRegression(class_weight='balanced', solver='saga', multi_class='multinomial')
    # lrmodel.fit(X, Y)
    return (lrmodel, feature_dict, tag_dict)
    pass



# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    pass


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    pass


# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    pass


def main(args):
    model, feature_dict, tag_dict = train(0.25)
    # print(get_features(['the', 'Happy', 'cAt-1'], 2, 'cAt-1'))

    # print(remove_rare_features([['a', 'e', 'b', 'c'], ['a', 'b', 'd']], 2))
    # print(remove_rare_features([[get_features(['the', 'happy', 'the', 'happy', 'happy', 'cat'], i, 'prev') for i in range(0, 6)]], 2))
    # Build feature and tag dictionaries
    # common_features is a set of strings
    # corpus_tags is a list of lists of strings (tags)
    # Returns a tuple (feature_dict, tag_dict)
    # print(get_feature_and_label_dictionaries(['a', 'b', 'c', 'd'], [['tag1', 'tag2', 'tag1'], ['tag4', 'tag3']]))
    # print(build_X([['a', 'b', 'c'], ['b', 'a', 'd']], {'a': 1, 'b': 2, 'c': 3, 'd': 4}))
    # predictions = predict('test.txt', model, feature_dict, tag_dict)
    # for test_sent in predictions:
    #     print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
