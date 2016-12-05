#NOTE: Make update just update values at a given parameter rather than do so much shit

from os import listdir, stat
from os.path import isfile, join
from enamexparser import *
class NaiveBayesClassifier():
    
    # probs is a dictionary of dictionaries of counters?
    def __init__(self, numOfFeatures, probs={}, classes=[]):
        self.probs = probs
        self.probsCalculated = False
        self.totalCount = 0
        self.classes = classes
        self.DEFAULT_PROB = 0.01
        for i in range(numOfFeatures):
            self.probs[i] = {}
        for c in classes:
            self.probs[c] = 0
        
    # Calculate the most probable type from probabilities
    def argmax(self, probability):
        highestT = None
        highestP = 0
        for (t, p) in probability.items():
            if p > highestP:
                highestP = p
                highestT = t
        return highestT

    # Classify a feature set
    def classify(self, features):
        if(not self.probsCalculated):
            self.calcProbs()
        probability = {}
        for c in self.classes:
            probability[c] = self.finalProbs[c]

        for (k, v) in features.items():
            try:
                if v == None:
                    raise KeyError
                probDict = {}
                probDict = self.finalProbs[k][v]
                for c in self.classes:
                    prob = probDict.get(c)
                    if prob == None or prob == 0:
                        prob = self.DEFAULT_PROB
                    probability[c] *= prob
            except KeyError:
                continue
            
        return probability

    # Update the values of the classifier
    def update(self, f, v, c, justF=False):
        if(justF):
            self.probs[c] += 1
            self.totalCount += 1
        else:
            try: 
                self.probs[f][v][c] += 1
            except KeyError:
                try:
                    self.probs[f][v][c] = 1
                except KeyError:
                    self.probs[f][v] = {}
                    self.probs[f][v][c] = 1

    # Calculate the final probabilities
    def calcProbs(self):
        finalProbs = {}
        for (k, v) in self.probs.items():
            if(type(v) is dict):
                finalProbs[k] = {}
                for (ik, iv) in v.items():
                    finalProbs[k][ik] = {}
                    count = 0
                    for c in self.classes:
                        iiv = iv.get(c)
                        if(iiv != None):
                            count += iiv
                    for c in self.classes:
                        iiv = iv.get(c)
                        if(iiv == None or iiv == 0):
                            finalProbs[k][ik][c] = None
                        else:
                            finalProbs[k][ik][c] = iiv / count
        for c in self.classes:
            # If we see a feature set with absolutely no features we've seen before, we have no clue
            if self.totalCount != 0:
                finalProbs[c] = self.probs[c] / self.totalCount
            else:
                finalProbs[c] = self.DEFAULT_PROB
        
        self.probsCalculated = True
        self.finalProbs = finalProbs

