from os import listdir, stat
from os.path import isfile, join

from enamexparser import *
from enamex_bayes_classifier import *
from test import *

import nltk
from nltk.corpus import PlaintextCorpusReader

import wikification

mypath="/home/brendan/work/nlp/ner/wsj_testdata"

### NERTagger is the class capable of training from, tagging and putting together files
class NERTagger():

    # Create a new tagger
    def __init__(self):
        self.parser = None
        self.classifier = EnamexBayesClassifier()
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
        self.NAMES = NAMES

    # This method will tag files from in a given path
    # It returns a dictionary of files which has values of their tagged parses,
    # as well as a similar dictionary of all named entities found in a given file.
    def tag(self, path, files):
        corpus = PlaintextCorpusReader(path, files)
        count = 1
        taggedDocs = {}
        nes = {}
        for f in files:
            print("Tagging file " + f + ", " + str(count) + "/" + str(len(files)))
            count += 1

            chunked = self.process(corpus.raw(f))
            for i in range(len(chunked)):
                for j in range(len(chunked[i])):
                    chunked[i][j] = self.neTagSentence(chunked[i][j])
            taggedDocs[f] = chunked

            foundNes = []
            for split in chunked:
                for sentence in split:
                    foundNes += self.getNesFromTree(sentence)
            nes[f] = foundNes
        return (taggedDocs, nes)
             
    # This method will train the Naive Bayes classifier.
    # All that needs to be provided is the training directory and a list of training file names.
    def train(self, trainDir, trainFiles, parseFiles=True):
        if parseFiles:
            self.parser = EnamexParser(trainDir, trainFiles)
        count = 0
        for f in trainFiles:
            count += 1
            parsed = self.parser.get(f)
            print("Training: " + f + " - " + str(count)+"/"+str(len(trainFiles)))
            self.classifier.train(EnamexParser.neTreesFromParsed(parsed))

    # This method can be used to test the system.
    # trainDir - the training file directory
    # trainFiles - the training files list
    # testDir - the test file directory
    # testFiles - the test file list
    # looseTest - whether a loose test should be performed or not (boolean)
    def test(self, trainDir, trainFiles, testDir, testFiles, looseTest=False):
        print("NOTE: The following training has no effect on the tag function. It is merely for testing.")
        print("The test directory MUST also contain tagged versions of the training files for evaluation.")

        oldClassifier = self.classifier
        oldParser = self.parser
        self.classifier = EnamexBayesClassifier()
        self.parser = EnamexParser(trainDir, (trainFiles+testFiles))

        #self.train(trainDir, trainFiles, False)
        (taggedDocs, nes) = self.tag(testDir, testFiles)

        percSum = 0
        ignored = 0
        for f in testFiles:
            perc = 0
            res = None
            if looseTest:
                res = Test.looseCalculate(nes[f], EnamexParser.nesFromParsed(self.parser.get(f)))
                if not res == None:
                    perc = 100 * res
            else:
                res = Test.calculate(nes[f], EnamexParser.nesFromParsed(self.parser.get(f)))
                if not res == None:
                    perc = 100 * res
            if res == None:
                ignored += 1
            else:
                percSum += perc
                print("Test: " + f + " " + str(perc) + "%")

        self.classifier = oldClassifier
        self.parser = oldParser

        print("Average: " + str(percSum / (len(testFiles) - ignored)) + "%")
        print("Achieved using " + str(len(trainFiles)) + " training files and " + str(len(testFiles)) + " test files.")
        
    # Process raw text, carries out tokenization and chunking, as well as some simple replacements.
    def process(self, raw):
        sents = []
        DATES = ["december", "dec.", "dec",
                 "november", "nov.", "nov",
                 "october",  "oct.", "oct",
                 "september", "sep.", "sep",
                 "august", "aug.", "aug",
                 "july", "jul.", "jul",
                 "june", "jun.", "jun",
                 "may",
                 "april", "apr.", "apr",
                 "march", "mar.", "mar",
                 "february", "feb.", "feb",
                 "january", "jan.", "jan",
                 "monday", "mon." "mon",
                 "tuesday", "tu.", "tue.", "tues",
                 "wednesday", "wed.",
                 "thursday", "th.", "thu.", "thur.", "thurs",
                 "friday", "fri." "fri",
                 "saturday", "sat.",
                 "sunday", "sun."]
        
        # Split at new lines since otherwise things such as Ltd.\nBob are the same sentence
        for s in raw.split("\n"):
            sents.append(nltk.sent_tokenize(s))
        for i in range(len(sents)):
            for j in range(len(sents[i])):
                sents[i][j] = nltk.pos_tag(nltk.word_tokenize(sents[i][j]))
                for k in range(len(sents[i][j])):
                    if sents[i][j][k][0] == "&":
                        sents[i][j][k] = (sents[i][j][k][0], "AMP")
                    elif sents[i][j][k][0] == ".":
                        sents[i][j][k] = (sents[i][j][k][0], "PERIOD")
                    elif (sents[i][j][k][0].lower()) in DATES:
                        sents[i][j][k] = (sents[i][j][k][0], "DATE")
                    elif sents[i][j][k][0] in self.NAMES:
                        sents[i][j][k] = (sents[i][j][k][0], "NNP")
                    elif not k == 0:
                        if sents[i][j][k][0][0].isupper():
                            sents[i][j][k] = (sents[i][j][k][0], "NNP")
                    elif sents[i][j][k][1] == "CD" and ":" in sents[i][j][k][0]:
                        sents[i][j][k] = (sents[i][j][k][0], "TIME")
                sents[i][j] = self.chunk(sents[i][j])

        return sents

    # Turn the result of a value of taggedDocs from the tag method into a text document.
    def stitch(self, tagged):

        sentences = list(tagged)

        for i in range(len(sentences)):
            sentences[i] = self.flatten(sentences[i])
        sentences = "\n".join(sentences)
        endOfSentPunct = [")","%",",",".","!","?",":",";"]
        count = 0

        sentences = list(sentences)
        i = 0 
        while i < (len(sentences)-1):
            if(sentences[i] == " "):
                if(sentences[i+1] in endOfSentPunct):
                    sentences[i] = sentences[i+1]
                    sentences.pop(i+1)
            i+=1

        endOfSentPunct = ["`", "£", "(", "$"] 
        i=1
        while i < (len(sentences)):
            if(sentences[i] == " "):
                if(sentences[i-1] in endOfSentPunct):
                    sentences[i] = sentences[i-1]
                    sentences.pop(i-1)
            i+=1
        
        sentences = self.fixQuotes(sentences)

        return ''.join(sentences)

    # Attempt to fix the quotes in a text
    def fixQuotes(self, text):
        seenSingle = False
        seenDouble = False

        i = 0
        while i  < len(text):
            if text[i] == "\'":
                text[i] = "'"
                if seenSingle:
                    if text[i-1] == " " and i-1 >= 0:
                        text.pop(i-1)
                        i-=1
                    seenSingle = False
                else:
                    if i+1 < len(text):
                        if text[i+1] == " ":
                            text.pop(i+1)
                    seenSingle = True
            elif text[i] == '\"':
                text[i] = '"'
                if seenDouble:
                    if text[i-1] == " " and i-1 >= 0:
                        text.pop(i-1)
                        i-=1
                    seenDouble = False
                else:
                    if i+1 < len(text):
                        if text[i+1] == " ":
                            text.pop(i+1)
                    seenDouble = True 

            i+=1

        return text


    # Carry out a shallow parse on the POS tagged data
    def chunk(self, posTagged):

        grammar = """
        NE: {<NNP.?>+}
        NE: {(<NNP.?>|<NE>)<PERIOD><NE>}
        NE: {<NE><AMP><NE>}
        NE: {<NE><POS><NE>}
        NE: {<NE><IN><NE>}
        NE: {(<CD><NE>|<NE><CD>)}
        """

        cp = nltk.RegexpParser(grammar)

        parsed = cp.parse(posTagged)

        return parsed

    # Convert a named entity to a tag
    def neToTag(self, label, t):
        res = "<ENAMEX TYPE=\"" + label + "\">"
        for c in t: 
            res += c + " "
        return (res[:len(res)-1] + "</ENAMEX>")

    # Extract a list of the named entity words
    def extractWordTag(self, t):
        r = []
        for i in t:
            if type(i) is nltk.Tree:
                for w in self.extractWordTag(i):
                    r.append(w)
            else:
                r.append(i)
        return r

    # Turn a tree into a phrase
    def flatten(self, t):
        phrase = ""
        for c in t:
            if(type(c) is nltk.Tree):
                classes = ["ORGANIZATION","LOCATION","PERSON"]
                if(c.label() in classes):
                    phrase += " " + self.neToTag(c.label(), c)    
                else:
                    phrase += " " + self.flatten(c)
            else:
                if c[1] == "POS":
                    phrase += c[0]
                else:
                    phrase += " " + c[0]
        return phrase[1:]

    # Get the first word of a tree
    def getFirstWord(self, t):
        if type(t) is nltk.Tree:
            return self.getFirstWord(t[0])
        else:
            return t

    # Tag a sentence
    def neTagSentence(self, s):
        for i in range(len(s)):
            if type(s[i]) is nltk.Tree:
                if(s[i].label() == "NE"):
                    lastNe = None
                    nextNe = None
                    tagMultiple = False
                    for c in s[i]:
                        if type(c) is nltk.Tree:
                            if lastNe == None:
                                lastNe = c
                            else:
                                nextNe = c
                        else:
                            if c[1] == "IN":
                                tagMultiple = True 
                    flattened = self.flatten(s[i])
                    result = self.classify(flattened, s, i)
                    s[i] = nltk.Tree(result, [flattened])

        return s
        
                    
    # Classify a named entity 
    def classify(self, ne, s, i):
        locRE = ["\d+\s[A-Za-z]+", "[A-Za-z]+\s\d+"]
        for r in locRE:
            if re.match(r, ne.lower()):
                return "LOCATION"

        orgRE = [".*inc.*", ".*corp.*", ".*ltd.*"]
        for r in orgRE:
            if re.match(r, ne.lower()):
                return "ORGANIZATION"

        titles = ["mr","prof","sir","mrs","ms","professor"]
        lowered = ne.lower()
        for t in titles:
            if (t+".") in lowered or (t+" ") in lowered:
                return "PERSON"

        result = wikification.wikiLookup(ne)

        if result == None:
            if i+1 >= len(s):
                result = self.classifier.classifyNE(self.extractWordTag(s[i]), None)
            else:
                w = self.getFirstWord(s[i+1])
                result = self.classifier.classifyNE(self.extractWordTag(s[i]), w)

        return result

    # Get all the named entities from a tree
    def getNesFromTree(self, tree):
        nes = []
        for e in tree:
            if type(e) is nltk.Tree:
                if(e.label() != "S"):
                    ne = ""
                    for w in e:
                        ne += " " + w
                    nes.append((e.label(), ne[1:]))
        return nes


###
# For an example of testing, uncomment the following:
#
#path = "./wsj_test_tagged/"
#allFiles = [f for f in listdir(path) if isfile(join(path, f))]
#nerTagger = NERTagger()
#nerTagger.test("./wsj_test_tagged/", allFiles[:1400], "./wsj_testdata", allFiles[1400:1405], False)
###

###
# For an example of training, tagging one file and then stitching together, uncomment the following:
#
#path = "./wsj_test_tagged/"
#allFiles = [f for f in listdir(path) if isfile(join(path, f))]
#nerTagger = NERTagger()
#nerTagger.train("./wsj_test_tagged/", allFiles[:1400])
#(tagged, ne) = nerTagger.tag("./wsj_testdata/", allFiles[1400:1401])
#for s in tagged.keys():
#    print(nerTagger.stitch(tagged[s]))
###
