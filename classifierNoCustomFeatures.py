# -*- coding: utf-8 -*-
# Code adapted from Scott Spurlock in-class lectures 10/21/15
# Some code taken from: 
# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py
# The rest implemented by Daniel Schneider 1/17/2016
#
# This program creates a classifier which identifies emails as written by Linus Torvalds or Greg Kroah-Hartman
# --------------------------------------------------------------------------
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pylab as plt
from time import time
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import random
import re
import operator
from sklearn import preprocessing
from scipy import sparse
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize
import math

#Initialize multiple variables for this program
#numtests is the number of iterations the classifier will run
#overallaccuracy is used to compute the accuracy of the classifier for 1 or more runs
#avgTruePositives,avgFalseNegative,avgFalsePositive,avgTrueNegative are variables 
#that build a confusion matrix overall for the number of iterations the classifier ran
#averageAllTerms computes the average score for each of the most discerning words computed by the classifier
#avgAllRatios computes the Linus:Greg ratio for each of the most discerning words computed by the classifier
numTests= 10
overallAccuracy=0
avgTruePositives=0
avgFalseNegative=0
avgFalsePositive=0
avgTrueNegative=0
avgPrecision=0
avgRecall=0
avgAllTerms={}
avgAllRatios={}

#Computes the Linus:Greg ratio for a supplied word
def findRatioForCertainWord(vectorizer,clf,word):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.feature_log_prob_[0],clf.feature_log_prob_[1], feature_names))
    for (myScore1,myScore2,myName) in coefs_with_fns:
        if myName ==word:
            logTransformedScore1=math.exp(myScore1)
            logTransformedScore2=math.exp(myScore2)
            return logTransformedScore1/(logTransformedScore1+logTransformedScore2)
            #print word + " linus:greg         " + str(logTransformedScore1/(logTransformedScore1+logTransformedScore2)) + " : " + str(logTransformedScore2/(logTransformedScore1+logTransformedScore2))                  

# Initialize posCorpus and negCorpus arrays
# These arrays will hold emails whether written by Greg (posCorpus) or Linus (negCorpus)
posCorpus = []
negCorpus = []

# read in emails from text files and assign them to the posCorpus for Greg
# change all text contents to lowercase
# remove all indications of Greg's name from emails as they usually identify a signature line at the end so would not be useful for classification
# remove all subsets of xx's between two x's and 22 x's as these indicate separators in the email between chunks of the email and are not useful for classification

inDir='C:/Users/danie_000/Documents/Elon/Big Honors Project of Death/Data/gregMoreStripping3/'
for filename in os.listdir(inDir):
    with open(os.path.join(inDir, filename)) as fh:
        contents = fh.read()
        contents=contents.lower()
        contents=contents.replace("greg","")
        contents= contents.replace("hartman","")
        contents= contents.replace("kroah","")
        contents= contents.replace("kh","")
        contents= contents.replace("'","")
        xsToRemove="xx"
        for i in range (2,20):
            contents= contents.replace(xsToRemove,"")
            xsToRemove=xsToRemove+"x"
        posCorpus.append(contents)
print "FINISHED GREG"

# read in emails from text files and assign them to the posCorpus for Linus
# change all text contents to lowercase
# remove all indications of Linus's name from emails as they usually identify a signature line at the end so would not be useful for classification
# remove all subsets of xx's between two x's and 22 x's as these indicate separators in the email between chunks of the email and are not useful for classification
inDir='C:/Users/danie_000/Documents/Elon/Big Honors Project of Death/Data/linusMoreStripping3/'
for filename in os.listdir(inDir):
    with open(os.path.join(inDir, filename)) as fh:
        contents = fh.read()
        contents=contents.lower()
        contents=contents.replace("linus","")
        contents= contents.replace("torvalds","")
        contents= contents.replace("'","")
        xsToRemove="xx"
        for i in range (2,20):
            contents= contents.replace(xsToRemove,"")
            xsToRemove=xsToRemove+"x"
        negCorpus.append(contents)
print "FINISHED LINUS"

#Read in a list of expletive words compiled by Google https://gist.github.com/jamiew/1112488
#Assign this list to a variable called expletiveList
expletiveList=[]
inDir='C:/Users/danie_000/Documents/Elon/Big Honors Project of Death/Python/Classifier'
with open(os.path.join(inDir, "googleExpletiveWords.txt")) as fh:
        contents = fh.read()
        contents=contents.lower()
        for word in contents.split():
            expletiveList.append(word)

#Run the classifier a pre-decided number of times
for x in range(0, numTests):
    # shuffle the positve and negative corpuses
    random.shuffle(posCorpus)
    random.shuffle(negCorpus)
    
    # assign 8/10 of Linus's emails and Greg's emails to the training set
    dataTrain = posCorpus[:len(posCorpus)*8/10] + negCorpus[:len(negCorpus)*8/10] # combined corpus
    
    #this portion of code initializes three lists which will be used as custom terms in the TFIDF classifier
    #dataTrainExpletiveCount is a count of the number of expletives for each of the email documents
    #dataTrainExclamationsCount is a count of the number of exclamations for each of the email documents
    #dataTrainAdverbCount is a count of the number of adverbs for each of the email documents
    dataTrainExpletiveCounts=[]
    dataTrainExclamationsCount = []
    dataTrainAdverbCount = []
    
    
    # set up a vector to hold the class label (0 or 1)
    y_train_pos = np.ones(len(posCorpus[:len(posCorpus)*8/10]))
    y_train_neg = np.zeros(len(negCorpus[:len(negCorpus)*8/10]))
    y_train = np.hstack((y_train_pos, y_train_neg))
    
    #CHOOSE VECTORIZER TO RUN WITH
    #create a TF-IDF vectorizers and train it with the training documents
    #vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    
    #comment previous vectorizer and uncomment next line to run code with frequency count vectors
    #vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                #stop_words='english')
                                
    #comment previous vectorizer and uncomment next line to run code with bigrams                        
    #vectorizer = TfidfVectorizer(min_df=1, ngram_range=(2,2))
    
    #comment previous vectorizer and uncomment next line to run code with bigrams                        
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(3,3))
    
    
    X_train = vectorizer.fit_transform(dataTrain)

   
    # create test group from 2/10 of the documents
    dataTest = posCorpus[len(posCorpus)*8/10:] + negCorpus[len(negCorpus)*8/10:]
    
    #this portion of code initializes three lists which will be used as custom terms in the TFIDF classifier
    #dataTrainExpletiveCount is a count of the number of expletives for each of the email documents
    #dataTrainExclamationsCount is a count of the number of exclamations for each of the email documents
    #dataTrainAdverbCount is a count of the number of adverbs for each of the email documents
    dataTestExpletiveCounts=[]
    dataTestExclamationsCounts=[]
    dataTestAdverbCount = []
    
    # set up a vector to hold the class label (0 or 1)
    y_test_pos = np.ones(len(posCorpus[len(posCorpus)*8/10:]))
    y_test_neg = np.zeros(len(negCorpus[len(negCorpus)*8/10:]))
    y_test = np.hstack((y_test_pos, y_test_neg))
    
    # each row is a test document, each column is a tfidf score
    X_test = vectorizer.transform(dataTest)
    
    #create a multinomial naive bayes classifier and fit it with the training data
    #predict the test data group based on the training
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_test_pred = nb.predict(X_test)
    y_test_probs = nb.predict_proba(X_test)
    
    # Compare predicted class labels with actual class labels
    accuracy=nb.score(X_test,y_test)
    
    #Add this accuracy to overall accuracy to be averaged later
    overallAccuracy=overallAccuracy+accuracy
    
    #Add precision and recall to overall sums to be averaged later
    avgPrecision= avgPrecision + precision_recall_fscore_support(y_test, y_test_pred, average='binary')[0]
    avgRecall = avgRecall+precision_recall_fscore_support(y_test, y_test_pred, average='binary')[1]
    
    #construct a confusion matrix and store the measures as sums to be averaged later
    nbcm = confusion_matrix(y_test, y_test_pred)
    avgTruePositives=avgTruePositives+nbcm[0][0]
    avgFalseNegative=avgFalseNegative+nbcm[0][1]
    avgFalsePositive=avgFalsePositive+nbcm[1][0]
    avgTrueNegative=avgTrueNegative+nbcm[1][1]
    
    #calculate the top 100 most influential features and put them in a dictionary with their score
    nFeats = 100
    ch2 = SelectKBest(chi2, k=nFeats)
    X_train_2 = ch2.fit_transform(X_train, y_train)
    X_test_2 = ch2.transform(X_test)
    
    #find all feature_names and add the custom feature names to this list
    feature_names = vectorizer.get_feature_names()
    
    #loop through the computed most influential features and determine their score and ratios
    #add scores to a dictionary called avgAllTerms where the name of the feature is the key
    #and the sum of the scores as well as the number of times this term appeared in each of the classifier iterations is the value
    #add ratios to a dictionary called avgAllRatios where the name of the feature is the key
    #and the sum of the ratios as well as the number of times this term appeared in each of the classifier iterations is the value
    #the sum of the scores and ratios will be averaged with the number of times the word appeared as a classifier term later in the program
    #Note: the custom features are ignored for ratios as they are not part of the original vectorizer component, so their ratios cannot be computed
    for i in ch2.get_support(indices=True):
        if feature_names[i] in avgAllTerms.keys():
            avgAllTerms[feature_names[i]]=(avgAllTerms[feature_names[i]][0]+ch2.scores_[i],avgAllTerms[feature_names[i]][1]+1)
        else:
            avgAllTerms[feature_names[i]]=(ch2.scores_[i],1)
        if feature_names[i]!='expletivesCount' and feature_names[i]!='exclamationsCount' and feature_names[i]!='adverbCount':
            if feature_names[i] in avgAllRatios.keys():
                avgAllRatios[feature_names[i]]=(avgAllRatios[feature_names[i]][0]+findRatioForCertainWord(vectorizer,nb,feature_names[i]),avgAllRatios[feature_names[i]][1]+1)
            else:
                avgAllRatios[feature_names[i]]=(findRatioForCertainWord(vectorizer,nb,feature_names[i]),1)
    print "Iteration "+ str(x+1) + " complete"
        
    

#Compute average measures by dividing by the number of times they were tested
overallAccuracy=overallAccuracy/numTests
avgTruePositives=float(avgTruePositives)/numTests
avgFalseNegative=float(avgFalseNegative)/numTests
avgFalsePositive=float(avgFalsePositive)/numTests
avgTrueNegative=float(avgTrueNegative)/numTests
avgPrecision=float(avgPrecision)/numTests
avgRecall=avgRecall/numTests

#compute average scores for all terms by dividing by the number of times they appeared in all of the iterations of classification
for word in avgAllTerms:
    avgAllTerms[word]=(avgAllTerms[word][0]/avgAllTerms[word][1],avgAllTerms[word][1])

#compute average ratios for all terms by dividing by the number of times they appeared in all of the iterations of classification
for ratio in avgAllRatios:
    avgAllRatios[ratio]=(avgAllRatios[ratio][0]/avgAllRatios[ratio][1],avgAllRatios[ratio][1])

#sort the score and ratio dictionaries in descending order
sorted_terms = sorted(avgAllTerms.items(), key=operator.itemgetter(1),reverse=True)
sorted_ratios = sorted(avgAllRatios.items(), key=operator.itemgetter(1),reverse=True)

#Print measures
print "NB Accuracy: %.10f" % overallAccuracy
print "NB True Positives: %.10f" % avgTruePositives
print "NB False Negatives: %.10f" %avgFalseNegative
print "NB False Positives: %.10f" %avgFalsePositive
print "NB True Negatives: %.10f" %avgTrueNegative
print "Precision: %.10f" % avgPrecision
print "Recall: %.10f" % avgRecall
print

print "Most important words:"
for word,score in sorted_terms:
    print word + ": " + str(score)
    
print "Most important ratios:"
for word,linusRatio in sorted_ratios:
    print word + "            linus:greg         " + str(linusRatio[0]) + ":"+ str(1-linusRatio[0])
    
print "Most important ratios sorted by importance of words:"
for word,score in sorted_terms:
    if word in avgAllRatios.keys():
        print word + "            linus:greg         " + str(avgAllRatios[word][0])
