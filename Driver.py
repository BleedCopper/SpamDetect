import math
import os
import random
from operator import itemgetter

from nltk import word_tokenize


# folder to list of emails
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    spam = 0
    ham = 0
    for a_file in file_list:
        f = open(folder + a_file, 'r')

        # all spam messages start with spmsg
        if a_file.startswith("spmsg"):
            a_list.append((f.read(), "spam"))
            spam += 1
        else:
            a_list.append((f.read(), "ham"))
            ham += 1
    f.close()
    return spam, ham, a_list


# return sentence into array of words
def preprocess(sentence):
    return word_tokenize(sentence)


# return True for every word in list
def get_features(text):
    return {word: True for word in preprocess(text)}


# train NB
def train(features_train):
    spamNum = 0
    voc = {}
    spamVoc = {}
    condprob = {}
    prior = 0

    priorh = 0
    hamVoc = {}
    condprobh = {}
    for email in features_train:
        # print(email[0])
        if (email[1] == "spam"):
            spamNum += 1
        for key, value in email[0].items():
            if key not in voc:
                voc[key] = 1
                spamVoc[key] = 0
                hamVoc[key] = 0
            else:
                voc[key] += 1
            if (email[1] == "spam"):
                spamVoc[key] += 1
            else:
                hamVoc[key] += 1

    prior = spamNum / len(features_train)
    priorh = (len(features_train) - spamNum) / len(features_train)

    # print(prior)
    for key, value in voc.items():
        # print(spamVoc[key],spamNum)
        condprob[key] = (spamVoc[key] + 1) / (spamNum + 2)
        condprobh[key] = (hamVoc[key] + 1) / ((len(features_train) - spamNum) + 2)

    # print(condprob)
    newVoc = mutualinf(voc, spamVoc, hamVoc, spamNum, len(features_train))
    return newVoc, prior, condprob, priorh, condprobh


def evaluate(test_set, voc, prior, condprob, priorh, condprobh, threshold):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0

    t = threshold / (1 + threshold)
    for email in test_set:
        score = (prior)
        scoreh = (priorh)
        for key in voc:
            if (key in email[0]):
                score *= (condprob[key])
                scoreh *= (condprobh[key])
            else:
                score *= (1-condprob[key])
                scoreh *= (1-condprobh[key])



        score = score / (score + scoreh)
        if (score > t):
            if (email[1] == "spam"):
                tp += 1
            else:
                fp += 1
        else:
            if (email[1] == "ham"):
                tn += 1
            else:
                fn += 1

    print(tp, fp, tn, fn)
    # acc = ((tp + (tn * threshold)) / (tp + ((tn + fp) * threshold) + fn))
    # recall = (tp / (tp + fn))
    # precision = (tp / (tp + (fp * threshold)))
    # err = ((fn + (fp * threshold)) / (tp + ((tn + fp) * threshold) + fn))
    acc = ((tp + (tn * threshold)) / (tp + ((tn + fp) * threshold) + fn))
    recall = (tp / (tp + fn))
    precision = (tp / (tp + (fp)))
    err = ((fn + (fp * threshold)) / (tp + ((tn + fp) * threshold) + fn))

    return acc, recall, precision, err


def mutualinf(voc, spamVoc, hamVoc, numSpam, numDoc):
    list = {}
    i = 0
    for key, value in voc.items():
        sum = 0

        # print(spamVoc[key],hamVoc[key])
        # print(numSpam - spamVoc[key],numDoc-numSpam - hamVoc[key])
        # print(numDoc)

        c0x0 = (numDoc-numSpam - hamVoc[key]) / numDoc
        c0x1 = hamVoc[key] / numDoc
        c1x0 = (numSpam - spamVoc[key]) / numDoc
        c1x1 = spamVoc[key] / numDoc

        c0 = c0x0 + c0x1
        c1 = c1x0 + c1x1
        x0 = c0x0 + c1x0
        x1 = c0x1 + c1x1

        #               c = 1                c = 0
        # x = 1      spamVoc[key]            hamVoc[key]
        # x = 0   numSpam - spamVoc[key]  numDoc-numSpam - hamVoc[key]

        # x=0; c=spam
        try:
            sum += ((c1x0)*math.log((c1x0/(c1*x0)),2))
        except:
            sum += 0
        # x=1; c=spam
        try:
            sum += ((c1x1)*math.log((c1x1/(c1*x1)),2))
        except:
            sum += 0
        # x=0; c=legit
        try:
            sum += ((c0x0)*math.log((c0x0/(c0*x0)),2))
        except:
            sum += 0
        # x=1; c=legit
        try:
            sum += ((c0x1)*math.log((c0x1/(c0*x1)),2))
        except:
            sum += 0
        list[key] = sum

    sort = sorted(list.items(), key=itemgetter(1), reverse=True)

    newVoc = []
    for i in range(200):
        newVoc.append(sort[i][0])
        # print(newVoc[i], sort[i][1], spamVoc[newVoc[i]], hamVoc[newVoc[i]])
        # print(newVoc2[i], sort2[i][1], spamVoc[newVoc2[i]], hamVoc[newVoc2[i]])

    print("FIRST", newVoc)

    return newVoc


types = ["bare", "lemm", "lemm_stop", "stop"]
att_num = [50, 50, 100, 100]

# One training and evaluation run for BARE dataset on lambda=1 using ten-fold cross-validation
threshold = 999  # lambda
accuracy_base = 0
accuracy = 0
tcr = 0
recall = 0
precision = 0
err_acc=0
error_base=0

# ten-fold
for i in range(10):
    train_list = []
    test_list = []
    spam = 0
    ham = 0
    file_list = os.listdir("dataset/bare")

    # get all parts
    for j, file in zip(range(10), file_list):
        # get one as test data
        if (i == j):
            a, b, test_list = init_lists("dataset/bare/" + file + "/")
            spam += a
            ham += b

        # the rest are training data
        else:
            a, b, list = init_lists("dataset/bare/" + file + "/")
            train_list.extend(list)

    # print(len(train_list), len(test_list), spam, ham)

    acc_base = (threshold * ham) / ((threshold * ham) + spam)
    err_base = spam / ((threshold * ham) + spam)

    # shuffles the lists
    random.shuffle(test_list)
    random.shuffle(train_list)

    # get the features per email
    test_features = [(get_features(email), label) for (email, label) in test_list]
    train_features = [(get_features(email), label) for (email, label) in train_list]

    # training
    voc, prior, condprob, priorh, condprobh = train(train_features)

    # evaluating
    acc, rec, prec, err = evaluate(test_features, voc, prior, condprob, priorh, condprobh, threshold)

    # evaluate(test_features, voc, prior, condprob, priorh, condprobh, threshold)
    #
    # if err > 0:
    tcr_temp = err_base / err
    # else:
    #     tcr_temp = 0

    # accumulate for the ten-fold
    accuracy_base += acc_base
    accuracy += acc
    precision += prec
    recall += rec
    tcr += tcr_temp
    err_acc+=err
    error_base+=err_base


    print(tcr_temp, tcr)
#
# # get average
print(recall / 10, precision / 10, accuracy / 10, accuracy_base / 10, tcr / 10, (error_base/err_acc))
