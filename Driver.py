import math
import os
import random
from operator import itemgetter

import itertools
from nltk import word_tokenize


# folder to list of emails
def extract_emails(folder):
    email_list = []
    file_list = os.listdir(folder)
    spam = 0
    ham = 0
    for file in file_list:
        f = open(folder + file, 'r')

        # all spam messages start with spmsg
        if file.startswith("spmsg"):
            email_list.append((f.read(), "spam"))
            spam += 1
        else:
            email_list.append((f.read(), "ham"))
            ham += 1
    f.close()
    return spam, ham, email_list


# return sentence into array of words
def preprocess(sentence):
    return word_tokenize(sentence)


# return True for every word in list
def get_features(text):
    return {word: True for word in preprocess(text)}


# train NB
# SEE: http://blog.datumbox.com/wp-content/uploads/2013/09/naive-bayes-maths9.png
def train(features_train, attribute_count):
    spam_count = 0
    vocabulary = {}
    spam_vocabulary = {}
    probability_spam = {}
    ham_vocabulary = {}
    probability_ham = {}

    # count number of spam/ham emails that words are present in
    for email in features_train:
        if (email[1] == "spam"):
            spam_count += 1
        for key, value in email[0].items():
            if key not in vocabulary:
                vocabulary[key] = 1
                spam_vocabulary[key] = 0
                ham_vocabulary[key] = 0
            else:
                vocabulary[key] += 1
            if (email[1] == "spam"):
                spam_vocabulary[key] += 1
            else:
                ham_vocabulary[key] += 1

    prior_spam = spam_count / len(features_train)
    prior_ham = (len(features_train) - spam_count) / len(features_train)

    # calculate probability of email being spam/ham given a specific word
    for key, value in vocabulary.items():
        probability_spam[key] = (spam_vocabulary[key] + 1) / (spam_count + 2)
        probability_ham[key] = (ham_vocabulary[key] + 1) / ((len(features_train) - spam_count) + 2)

    # only use vocabulary with top mutual infos
    top_vocabulary = mutualinf(vocabulary, spam_vocabulary, ham_vocabulary, spam_count, len(features_train),
                               attribute_count)
    return top_vocabulary, prior_spam, probability_spam, prior_ham, probability_ham


# return evaluation metrics results
# SEE: http://blog.datumbox.com/wp-content/uploads/2013/09/naive-bayes-maths9.png
def evaluate(test_set, vocabulary, prior_spam, probability_spam, prior_ham, probability_ham, threshold):
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0

    t = threshold / (1 + threshold)

    # calculate probability of an email being spam/ham
    for email in test_set:
        score_spam = math.log(prior_spam)
        score_ham = math.log(prior_ham)
        for key in vocabulary:
            if (key in email[0]):
                score_spam += math.log(probability_spam[key])
                score_ham += math.log(probability_ham[key])
            else:
                score_spam += math.log(1 - probability_spam[key])
                score_ham += math.log(1 - probability_ham[key])

        # convert from log probability to regular probability
        score_spam = math.exp(score_spam)
        score_ham = math.exp(score_ham)

        # normalize
        score_spam = score_spam / (score_spam + score_ham)

        if (score_spam > t):  # email predicted as spam
            if (email[1] == "spam"):
                true_positive += 1
            else:
                false_positive += 1
        else:  # email predicted as ham
            if (email[1] == "ham"):
                true_negative += 1
            else:
                false_negative += 1

    accuracy = ((true_positive + (true_negative * threshold)) / (
        true_positive + ((true_negative + false_positive) * threshold) + false_negative))
    recall = (true_positive / (true_positive + false_negative))
    precision = (true_positive / (true_positive + (false_positive)))
    error = ((false_negative + (false_positive * threshold)) / (
        true_positive + ((true_negative + false_positive) * threshold) + false_negative))

    return accuracy, recall, precision, error


# return words with top mutual info
# SEE: http://stats.stackexchange.com/questions/191604/how-to-calculate-mutual-information-from-frequencies
def mutualinf(vocabulary, spam_vocabulary, ham_vocabulary, spam_count, doc_count, attribute_count):
    word_information = {}
    i = 0
    for key, value in vocabulary.items():
        information = 0

        ham_absent = (doc_count - spam_count - ham_vocabulary[
            key]) / doc_count  # ratio: ham emails that dont have the word
        ham_present = ham_vocabulary[key] / doc_count  # ratio: ham emails that has the word
        spam_absent = (spam_count - spam_vocabulary[key]) / doc_count  # ratio: spam emails that dont have the word
        spam_present = spam_vocabulary[key] / doc_count  # ratio: spam emails that has the word

        ham = ham_absent + ham_present  # ratio: ham emails
        spam = spam_absent + spam_present  # ratio: spam emails
        absent = ham_absent + spam_absent  # ratio: emails without the word
        present = ham_present + spam_present  # ratio: emails that has the word

        # x=0; c=spam
        try:
            information += ((spam_absent) * math.log((spam_absent / (spam * absent)), 2))
        except:
            information += 0
        # x=1; c=spam
        try:
            information += ((spam_present) * math.log((spam_present / (spam * present)), 2))
        except:
            information += 0
        # x=0; c=legit
        try:
            information += ((ham_absent) * math.log((ham_absent / (ham * absent)), 2))
        except:
            information += 0
        # x=1; c=legit
        try:
            information += ((ham_present) * math.log((ham_present / (ham * present)), 2))
        except:
            information += 0
        word_information[key] = information

    # sort words by information score
    sort = sorted(word_information.items(), key=itemgetter(1), reverse=True)

    # only take top N words
    top_information = []
    for i in range(attribute_count):
        top_information.append(sort[i][0])

    return top_information


# train and evaluate given parameters
def naive_bayes(type, threshold, attribute_count, train_portion):
    accuracy_base_list = []
    accuracy_list = []
    tcr_list = []
    recall_list = []
    precision_list = []

    # ten-fold
    for i in range(10):
        train_list = []
        test_list = []
        spam_count = 0
        ham_count = 0
        file_list = os.listdir("dataset/" + type)

        # get all parts
        for j, file in zip(range(10), file_list):
            # get one as test data
            if (i == j):
                spam, ham, test_list = extract_emails("dataset/" + type + "/" + file + "/")
                spam_count += spam
                ham_count += ham

            # the rest are training data
            else:
                spam, ham, temp_list = extract_emails("dataset/" + type + "/" + file + "/")
                random.shuffle(temp_list)
                train_list.extend(itertools.islice(temp_list, 0, (int)(len(temp_list) * train_portion)))

        accuracy_base_list.append((threshold * ham_count) / ((threshold * ham_count) + spam_count))
        error_base = spam_count / ((threshold * ham_count) + spam_count)

        # shuffles the lists
        random.shuffle(test_list)
        random.shuffle(train_list)

        # get the features(words) per email
        test_features = [(get_features(email), label) for (email, label) in test_list]
        train_features = [(get_features(email), label) for (email, label) in train_list]

        # training
        vocabulary, prior_spam, probability_spam, prior_ham, probability_ham = train(train_features, attribute_count)

        # evaluating
        accuracy, recall, precision, error = evaluate(test_features, vocabulary, prior_spam, probability_spam,
                                                      prior_ham, probability_ham, threshold)

        accuracy_list.append(accuracy)
        tcr_list.append(error_base / error)
        recall_list.append(recall)
        precision_list.append(precision)

    print(tcr_list)
    print(sum(recall_list) / 10, sum(precision_list) / 10, sum(accuracy_list) / 10, sum(accuracy_base_list) / 10,
          sum(tcr_list) / 10)
    return sum(recall_list) / 10, sum(precision_list) / 10, sum(accuracy_list) / 10, sum(accuracy_base_list) / 10, sum(
        tcr_list) / 10


# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# MAIN PROGRAM

types = ["bare", "stop", "lemm", "lemm_stop"]
thresholds = [1, 9, 999]

# 1. Table 2
naive_bayes(types[0], 1, 50, 1)
naive_bayes(types[1], 1, 50, 1)
naive_bayes(types[2], 1, 100, 1)
naive_bayes(types[3], 1, 100, 1)
naive_bayes(types[0], 9, 200, 1)
naive_bayes(types[1], 9, 200, 1)
naive_bayes(types[2], 9, 100, 1)
naive_bayes(types[3], 9, 100, 1)
naive_bayes(types[0], 999, 200, 1)
naive_bayes(types[1], 999, 200, 1)
naive_bayes(types[2], 999, 300, 1)
naive_bayes(types[0], 999, 300, 1)

# 2. Figure 1 *not sure about attribute count
for type in types:
    naive_bayes(type, 1, 50, 1)

# 3. Figure 2 *not sure about attribute count
for type in types:
    naive_bayes(type, 9, 50, 1)

# 4. Figure 3 *not sure about attribute count
for type in types:
    naive_bayes(type, 999, 50, 1)

# 5. Figure 4
for type in types:
    for attribute_count in range(50, 701, 50):
        naive_bayes(type, 1, attribute_count, 1)

# 6. Figure 5
for type in types:
    for attribute_count in range(50, 701, 50):
        naive_bayes(type, 9, attribute_count, 1)

# 7. Figure 6
for type in types:
    for attribute_count in range(50, 701, 50):
        naive_bayes(type, 999, attribute_count, 1)

# 8. Figure 7
for train_portion in range(10, 101, 10):
    naive_bayes(types[3], 1, 100, train_portion / 100)
    naive_bayes(types[3], 9, 100, train_portion / 100)
    naive_bayes(types[3], 999, 300, train_portion / 100)
