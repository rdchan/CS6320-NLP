import re
import sys

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])

# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    corpus_file = open(corpus_path, "r")
    corpus = corpus_file.read()
    lines = corpus.split("\n")
    tokenized_lines = []
    for line in lines:
        if (len(line) != 0):
            tag_split = line.split("\t")
            # print('***** tag split *****')
            # print(tag_split)
            sentence = sent_tokenize(tag_split[0])
            # print('***sentence***')
            # print(type(sentence))
            # print(sentence)
            words = word_tokenize(sentence[0])
            # print('**now words***')
            # print(type(words))
            # print(words)
            tokenized_lines.append((words, int(tag_split[1])))
            # print((words, int(tag_split[1])))
    return tokenized_lines
    pass


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if (word in negation_words):
        return True
    if (word[-3:] == "n\'t"):
        return True
    return False
    pass


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    single_string = " ".join(snippet)
    # tagged = nltk.pos_tag(single_string)
    tagged = nltk.pos_tag(snippet)
    print(snippet)
    print(single_string)
    print(tagged)
    tagged_snippet = []
    currently_negated = False
    check_not_only = False
    for word in snippet:
        if (word == "not"):
            check_not_only = True
        if (check_not_only):
            check_not_only = False
            if (word == "only"):
                currently_negated = False
        if is_negation(word):
            currently_negated = True
            tagged_snippet.append(word)
        else:
            if (word in negation_enders):
                currently_negated = False
                tagged_snippet.append(word)
            else:
                if(currently_negated):
                    tagged_snippet.append("NOT_" + word)
                else:
                    tagged_snippet.append(word)

    return tagged_snippet

    pass


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    position_idx = 0
    dictionary = {}
    for item in corpus:
        snippet = item[0]
        for word in snippet:
            if not word in dictionary:
                dictionary[word] = position_idx
                position_idx = position_idx + 1
    return dictionary
    pass

# Converts a snippet into a feature vector
# snippet is a list of strings
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    vector = np.zeros(len(feature_dict))
    for word in snippet:
        vector[feature_dict[word]] = vector[feature_dict[word]] + 1
    return vector
    pass


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):

    # TODO: do
    # x is size n x d, where n is the number of snippets in the corpus (len(corpus))
    # and d is the number of features in the feature_dict (len(feature_dict))
    x = np.empty([len(corpus), len(feature_dict)]) # holds the training feature vectors
    # y is of size n as well
    y = np.empty(len(corpus)) # holds the training feature labels
    for idx, line in enumerate(corpus):
        # line is a tuple (snippet, label)
        # print("evalulating ", snippet, "and label", label, "with idx", idx)
        # print("evalulating ", line, "with idx", idx)
        x[idx,:] = vectorize_snippet(line[0], feature_dict)
        y[idx] = line[1]
    return (x, y)
    pass


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    X = X.astype(float)
    for column_idx in range(len(X.T)):
        max_value = -np.inf
        min_value = np.inf
        for value in X.T[column_idx]:
            max_value = max(max_value, value)
            min_value = min(min_value, value)
        # now replace each value v with \frac{v-min}{max-min}
        # print("the max is", max_value, "the min is", min_value)
        for value_idx in range(len(X.T[column_idx])):
            # X.T[column_idx][value_idx] = (X.T[column_idx][value_idx]-min_value)/(max_value-min_value)
            print("looking at", X.T[column_idx][value_idx] )
            # print((X.T[column_idx][value_idx]-min_value)/(max_value-min_value))
            if (max_value == min_value):
                # X.T[column_idx][value_idx] = (X.T[column_idx][value_idx]-min_value)/(max_value-min_value)
                # X[value_idx][column_idx] = 0
                X.T[column_idx][value_idx] = 0
            else:
                print(X.T[column_idx][value_idx])
                X.T[column_idx][value_idx] = float((1.0*X.T[column_idx][value_idx])-(1.0*min_value))/float((1.0*max_value)-(1.0*min_value))
                print(X.T[column_idx][value_idx])
                print(X[value_idx][column_idx])
            # print((X[value_idx][column_idx]-min_value)/(max_value-min_value))
    print(X)
    print(X.T)
    print("leaving normalize")
    X[:] = X[:]
    # WHY IT NO STICK
    pass


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    pass


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    pass


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    pass


def main(args):
    corpus = load_corpus('test.txt')
    # snip = ['ice', 'age', 'wo', "n't", 'drop', 'your', 'jaw', ',', 'but', 'it', 'will', 'warm', 'your', 'heart', ',', 'and', 'i', "'m", 'giving', 'it', 'a', 'strong', 'thumbs', 'up', '.']
    f_dict = get_feature_dictionary(corpus)
    # print(f_dict)
    # print(tag_negation(snip))
    vectorize_corpus(corpus, f_dict)
    # model, feature_dict = train('train.txt')
    X = np.array([[1,2,3], [4, 5, 6], [7, 8, 9]])
    print(X)
    normalize(X)
    print(X)


    # print(test(model, feature_dict, 'test.txt'))

    # weights = get_top_features(model, feature_dict)
    # for weight in weights:
    #     print(weight)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
