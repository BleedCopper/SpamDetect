import os
import random
from collections import Counter

import nltk
from nltk import word_tokenize
from nltk import NaiveBayesClassifier, classify
from numpy import unicode

#folder to list of emails
def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    spam=0
    ham=0
    for a_file in file_list:
        f = open(folder + a_file, 'r')

        #all spam messages start with spmsg
        if a_file.startswith("spmsg"):
            a_list.append((f.read(),"spam"))
            spam+=1
        else:
            a_list.append((f.read(),"ham"))
            ham+=1
    f.close()
    return spam,ham, a_list


#return sentence into array of words
def preprocess(sentence):
    return word_tokenize(sentence)

#return True for every word in list
def get_features(text):
        return {word: True for word in preprocess(text)}

#train NB
def train(features_train):
    classifier = NaiveBayesClassifier.train(features_train)
    return classifier

def evaluate(test_set, classifier, threshold):
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0

    t=threshold/(1+threshold)
    for email in test_set:
        dist = classifier.prob_classify(email[0])

        if(dist.prob("spam")>t):
            if(email[1]=="spam"):
                tp+=1
            else:
                fp+=1
        else:
            if(email[1]=="ham"):
                tn+=1
            else:
                fn+=1

    acc = ((tp+(tn*threshold))/(tp+((tn+fp)*threshold)+fn))
    recall = (tp/(tp+fn))
    precision = (tp/(tp+(fp*threshold)))
    err = ((fn+(fp*threshold))/(tp+((tn+fp)*threshold)+fn))

    print(tp, tn, fp, fn)
    return acc,recall,precision,err

        # for label in dist.samples():
        #     print("%s: %f" % (label, dist.prob(label)))


types = ["bare", "lemm", "lemm_stop", "stop"]
att_num = [50,50,100,100]

#One training and evaluation run for BARE dataset on lambda=1 using ten-fold cross-validation
threshold = 1 #lambda
accuracy_base = 0
accuracy = 0
tcr = 0
recall=0
precision=0

#ten-fold
for i in range(10):
    train_list = []
    test_list = []
    spam=0
    ham=0
    file_list = os.listdir("dataset/bare")

    #get all parts
    for j, file in zip(range(10),file_list):
        #get one as test data
        if(i==j):
            a,b,test_list = init_lists("dataset/bare/"+file+"/")
            spam+=a
            ham+=b

        #the rest are training data
        else:
            a,b,list=init_lists("dataset/bare/"+file+"/")
            train_list.extend(list)

    print(len(train_list), len(test_list),spam, ham)

    acc_base=(threshold*ham)/((threshold*ham)+spam)
    err_base=spam/((threshold*ham)+spam)

    #shuffles the lists
    random.shuffle(test_list)
    random.shuffle(train_list)

    #get the features per email
    test_features = [(get_features(email), label) for (email, label) in test_list]
    train_features = [(get_features(email), label) for (email, label) in train_list]

    #training
    classifier = train(train_features)

    #evaluating
    acc,rec,prec,err=evaluate(test_features, classifier, threshold)
    if err!=0: tcr_temp = err_base/err
    else: tcr_temp =0

    #accumulate for the ten-fold
    accuracy_base+=acc_base
    accuracy+=acc
    precision+=prec
    recall+=rec
    tcr+=tcr_temp

#get average
print(recall/10, precision/10, accuracy/10, accuracy_base/10, tcr/10)


