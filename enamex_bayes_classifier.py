from naive_bayes import NaiveBayesClassifier
import re
import nltk

class EnamexBayesClassifier(NaiveBayesClassifier):

    def __init__(self):
        NaiveBayesClassifier.__init__(self, 5, classes=["ORGANIZATION", "LOCATION", "PERSON"]) 
        NAMES = []
        f = open('./gazeteers/names.male','r').read().splitlines()
        for n in f:
            NAMES.append(n)
        f = open('./gazeteers/names.female','r').read().splitlines()
        for n in f:
            NAMES.append(n)
        f = open('./gazeteers/names.family','r').read().splitlines()
        for n in f:
            NAMES.append(n)
        self.names = NAMES

    def classifyNE(self, t, after):
        t.append(after)
        previousNeTag = None
        classCount = {"ORGANIZATION": 0, "LOCATION": 0, "PERSON": 0}
        for i in range(len(t)-1):
            probs = self.classify(self.getFeatures(t[i], t[i+1], previousNeTag))
            previousNeTag = self.argmax(probs)
            classCount[previousNeTag] += 1 

        #probs = self.classify(self.getFeatures(t))
        return max(classCount, key=classCount.get)

    def train(self, nes):
        for i in range(len(nes)):
            #self.update(None, None, nes[i][0], True)
            #f = self.getFeatures(nes[i][1])
            #for j in range(len(f)):
            #    self.update(j, f[j], nes[i][0])
            previousNeTag = None
            self.update(None, None, nes[i].label(), True)
            words = nes[i][0]
            words.append(nes[i][1])
            for w in range(len(words)-1):
                f = self.getFeatures(words[w], words[w+1], previousNeTag)
                for j in range(len(f)):
                    self.update(j, f[j], nes[i].label())
                previousNeTag = nes[i].label()

                #if(i > 0):
                #    self.update(4, parsed[i-1][1], parsed[i][1])
             
                #### current token
                #self.update(0, parsed[i][0], parsed[i][1])
                #self.update(2, parsed[i][2], parsed[i][1])

                #### next token
                #if(i < (len(parsed)-1)):
                #    self.update(2, parsed[i+1][0], parsed[i][1])

                #    self.update(3, parsed[i+1][2], parsed[i][1])

                #self.update(5, EnamexBayesClassifier.persony(self.names, parsed[i][0]), parsed[i][1])
                #self.update(6, EnamexBayesClassifier.organizationy(parsed[i][0]), parsed[i][1])
                #self.update(7, EnamexBayesClassifier.locationy(parsed[i][0]), parsed[i][1])

    #def getFeatures(self, ne):
    #    f = {}
    #    tokens = nltk.word_tokenize(ne)
    #    f[0] = len(tokens)
    #    f[1] = sum (1 for i in ne if not (i < 'z' and i > 'a')  and not (i < 'Z' and i > 'A'))
    #    f[2] = sum (1 for i in ne)
    #    f[3] = self.persony(self.names, tokens)
    #    return f


    def getFeatures(self, ne, after, previousNeTag):
        f = {}
        f[0] = self.locationy(ne[0])
        f[1] = self.persony(self.names, ne[0])
        f[2] = ne[0]
        if(after == None):
            f[3] = None
        else:
            f[3] = after[0]
        if(previousNeTag == None):
            f[4] = None
        else:
            f[4] = previousNeTag
        
        return f

    #def persony(self, names, words):
    #    for w in words:
    #        if w in names:
    #            return True
    #    return False

    def persony(self, names, w, dontRecurse=False):

        if(dontRecurse):
            return w in names
        return (w in names) and not self.organizationy(w, True) and not self.locationy(w, True)

    def locationy(self, words):
        locWords = ["city", "united", "states", "road", "avenue","south","west","north","east"]
        for w in words:
            if word.lower() in locWords:
                return True
            if re.match(r"\d+", word):
                return True
            if re.match(r"(.\.)+", word):
                return True
        return False


    def organizationy(self, word, dontRecurse=False):
        orgWords = ["corp", "inc.", "company", "org", "ltd"]
        r = False
        for o in orgWords:
            r |= o in word.lower()
        if(dontRecurse):
            return r
        return r and not self.persony(self.names, word, True) and not self.locationy(word, True)

    def locationy(self, word, dontRecurse=False):
        locWords = ["city", "united", "states", "road", "avenue","south","west","north","east"]
        if word.lower() in locWords:
            if(dontRecurse):
                return True
            return not self.persony(self.names, word, True) and not self.organizationy(word, True)
        if re.match(r"\d+", word):
            if(dontRecurse):
                return True
            return not self.persony(self.names, word, True) and not self.organizationy(word, True)
        if re.match(r"(.\.)+", word):
            if(dontRecurse):
                return True
            return not self.persony(self.names, word, True) and not self.organizationy(word, True)

        return False




