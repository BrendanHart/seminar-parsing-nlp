import nltk, string, re

from ner import *

from os import listdir
from os.path import isfile, join
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from collections import Counter

path = "./test_untagged"

class Seminar():
    def __init__(self, topic=None, _type=None, speaker=None, location=None, startTime=None, endTime=None, tokenCount=None, tagged=None):
        self.topic = topic
        self._type = _type
        self.speaker = speaker 
        self.location = location
        self.startTime = startTime
        self.endTime = endTime
        self.topic = topic
        self.tokenCount = tokenCount
        self.tagged = tagged

    def compareLooseTime(time1, time2):
        if time1 == None and time2 == None:
            return True
        if time1 == None:
            return False
        if time2 == None:
            return False
        time1 = re.sub("[^0-9]", "", time1)
        time1 = time1.strip("0")
        time2 = re.sub("[^0-9]", "", time2)
        time2 = time2.strip("0")
        return time1 == time2

    def makeSeminarFromFile(path):
        newSeminar = Seminar()
        seminar = open(path).read()

        speaker = re.search("<speaker>(.+)</speaker>", seminar)
        if speaker:
            newSeminar.speaker = speaker.group(1)
            newSeminar.speaker = newSeminar.speaker.replace("<speaker>", "")
            newSeminar.speaker = newSeminar.speaker.replace("</speaker>", "")
            newSeminar.speaker = newSeminar.speaker.replace("<sentence>", "")
            newSeminar.speaker = newSeminar.speaker.replace("</sentence>", "")

        startTime = re.search("<stime>(.+)</stime>", seminar)
        if startTime:
            newSeminar.startTime = startTime.group(1)
            newSeminar.startTime = newSeminar.startTime.replace("<stime>", "")
            newSeminar.startTime = newSeminar.startTime.replace("</stime>", "")
            newSeminar.startTime = newSeminar.startTime.replace("<sentence>", "")
            newSeminar.startTime = newSeminar.startTime.replace("</sentence>", "")

        endTime = re.search("<etime>(.+)</etime>", seminar)
        if endTime:
            newSeminar.endTime = endTime.group(1)
            newSeminar.endTime = newSeminar.endTime.replace("<etime>", "")
            newSeminar.endTime = newSeminar.endTime.replace("</etime>", "")
            newSeminar.endTime = newSeminar.endTime.replace("<sentence>", "")
            newSeminar.endTime = newSeminar.endTime.replace("</sentence>", "")

        location = re.search("<location>(.+)</location>", seminar)
        if location:
            newSeminar.location = location.group(1)
            newSeminar.location = newSeminar.location.replace("<location>", "")
            newSeminar.location = newSeminar.location.replace("</location>", "")
            newSeminar.location = newSeminar.location.replace("<sentence>", "")
            newSeminar.location = newSeminar.location.replace("</sentence>", "")

        return newSeminar

            

class SeminarTagger():

    def __init__(self, tagger):
        self.nerTagger = tagger

    def tag(self, path, f):
        corpus = PlaintextCorpusReader(path, [f])
        result = {}

        #newSeminar = Seminar(f[:-4])
        newSeminar = Seminar()

        if(corpus.raw(f) == ""):
            result[f] = None
            return None

        (header, abstract) = self.extractHeaderAbstract(corpus.raw(f))
        orig = abstract
        headerDict = self.headerToDict(header)
        newSeminar._type = headerDict["Type"][0]

        chunked = self.nerTagger.process(abstract)
        
        # Try extract any times from header
        if "Time" in headerDict:
            time = re.findall("(\d{1,2}(:\d{1,2}(\s*[AaPp]\.?[Mm])?|\s*[AaPp]\.?[Mm]))", "\n".join(headerDict["Time"]))
            if len(time) == 2:
                newSeminar.startTime = time[0][0]
                newSeminar.endTime = time[1][0]
            else:
                newSeminar.startTime = time[0][0]

        if newSeminar.endTime == None:
            for i in range(len(chunked)):
                for j in range(len(chunked[i])):
                    times = re.findall("(\d{1,2}(:\d{1,2}(\s*[AaPp]\.?[Mm])?|\s*[AaPp]\.?[Mm]))", nerTagger.flatten(chunked[i][j]))
                    foundStart = False
                    for t in times:
                        if Seminar.compareLooseTime(t[0], newSeminar.startTime):
                            foundStart = True
                        else:
                            if foundStart:
                                newSeminar.endTime = t[0]
                                break
                    if newSeminar.endTime:
                        break
                if newSeminar.endTime:
                    break


        # Try extract speaker from header
        foundPerson = self.searchHeader(headerDict, "Who", "PERSON")
        firstFound = None
        if not foundPerson:
            # Try find speaker in abstract
            speakerWords = ["by", "speaker", "talk", "giving", "present"]
            (foundPerson, firstFound) = self.searchAbstract(chunked, speakerWords, "PERSON")
        if not foundPerson:
            # If we found no speakers and there's Who in the header, use the first phrase
            if "Who" in headerDict:
                firstLine = headerDict["Who"][0]
                firstLine = firstLine.split(",")
                firstLine = firstLine[0].split("-")
                foundPerson = firstLine[0].strip(" ")
        if not foundPerson:
            # Use the first found speaker if none were found in an expected sentence
            foundPerson = firstFound

        # Try extract location from header
        foundLocation = self.searchHeader(headerDict, "Place", "LOCATION")
        firstFound = None
        if not foundLocation and "Place" in headerDict:
            if len(headerDict["Place"]) == 1:
                foundLocation = headerDict["Place"][0].strip(" ")
        if not foundLocation:
            # Try find location in abstract
            locationWords = ["in", "at", "where"]
            (foundLocation, firstFound) = self.searchAbstract(chunked, locationWords, "LOCATION")
        if not foundLocation:
            # Use the first found location if none were found in an expected sentence
            foundLocation = firstFound
        if not foundLocation:
            # Default to all of "Place:"
            if "Place" in headerDict:
                foundLocation = headerDict["Place"][0].strip(" ")


        text = ""
        sents = []
        for s in abstract.split("\n\n"):
            sents.append(nltk.sent_tokenize(s))
        for i in range(len(sents)):
            first = True
            for j in range(len(sents[i])):
                if sents[i][j].startswith("\t\t"):
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                elif sents[i][j].startswith("  "):
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                elif sents[i][j].startswith("\n"):
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                elif sents[i][j].isupper():
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                elif re.match("(\s+|\t+)?[A-Z][A-Za-z]*:", sents[i][j].strip(" ")):
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                elif str.istitle(sents[i][j]):
                    text += sents[i][j]
                    sents[i][j] = (0, sents[i][j])
                else:
                    if first:
                        text += "<paragraph>"
                        first = False
                    text += "<sentence>" + sents[i][j] + "</sentence>"
                    sents[i][j] = (1, sents[i][j])
            if not first:
                text += "</paragraph>"
            text += "\n\n"
        abstract = text 

        # Replace the speaker in the text
        if foundPerson:
            foundPerson = foundPerson.strip(" ")
            foundPerson = re.sub("\(.+\)", "", foundPerson)
            newSeminar.speaker = foundPerson
            header = header.replace(newSeminar.speaker, ("<speaker>"+newSeminar.speaker+"</speaker>"))
            abstract = abstract.replace(newSeminar.speaker, ("<speaker>"+newSeminar.speaker+"</speaker>"))

        # Replace the location in the text
        if foundLocation: 
            newSeminar.location = foundLocation
            header = header.replace(newSeminar.location, ("<location>"+newSeminar.location+"</location>"))
            abstract = abstract.replace(newSeminar.location, ("<location>"+newSeminar.location+"</location>"))

        # Replace the times in the header
        header = header.replace(newSeminar.startTime, ("<stime>"+newSeminar.startTime+"</stime>"))
        if(newSeminar.endTime != None):
            header = header.replace(newSeminar.endTime, ("<etime>"+newSeminar.endTime+"</etime>"))
        # Get the occurences of the times in abstract
        (startTimes, endTimes) = self.looseTimeReplace(abstract, newSeminar.startTime, newSeminar.endTime)
        # Tag them all
        for t in startTimes:
            abstract = abstract.replace(t, "<stime>"+t+"</stime>")
        for t in endTimes:
            abstract = abstract.replace(t, "<etime>"+t+"</etime>")
        
        finalSeminarData = header + "\nAbstract:\n" + abstract
        newSeminar.tagged  = finalSeminarData
        
        count = Counter()
        for t in nltk.word_tokenize(orig):
            if not t in stopwords.words("english") and not t in ["--",",",".","<",">","-"]:
                count[t] += 1  

        newSeminar.tokenCount = count
        return newSeminar

    
    def looseTimeReplace(self, text, sTime, eTime=None):

        sTime = re.sub("[^0-9]", "", sTime)
        sTime = sTime.strip("0")
        if eTime:
            eTime = re.sub("[^0-9]", "", eTime)
            eTime = eTime.strip("0")
        
        foundTimes = set()
        times = re.findall("(\d{1,2}((\s*(AM|PM|am|pm|A\.M|P\.M|a\.m|p\.m))|(AM|PM|am|pm|A\.M|P\.M|a\.m|p\.m)))", text)
        for t in times:
            foundTimes.add(t[0])
        times = re.findall("(\d{1,2}:\d{1,2}((\s*(AM|PM|am|pm|A\.M|P\.M|a\.m|p\.m))|(AM|PM|am|pm|A\.M|P\.M|a\.m|p\.m))?)", text)
        for t in times:
            foundTimes.add(t[0])

        startTimes = set()
        endTimes = set()
        for t in foundTimes:
            compT = re.sub("[^0-9]", "", t)
            compT = compT.strip("0")
            if compT == eTime:
                endTimes.add(t)
            elif compT == sTime:
                startTimes.add(t)
        return (startTimes, endTimes)
            
    def searchAbstract(self, chunked, words, label):
        found = None
        firstFound = None
        for i in range(len(chunked)):
            for j in range(len(chunked[i])):
                nes = self.nerTagger.getNesFromTree(
                        self.nerTagger.neTagSentence(chunked[i][j]))
                for (l, ne) in nes:
                    flattened = self.nerTagger.flatten(chunked[i][j])
                    if l == label and (re.match("^\s*[A-Za-z]+:", flattened) or self.nerTagger.flatten(chunked[i][j]) == ('<ENAMEX TYPE="'+l+'">'+ne+"</ENAMEX>")):
                        found = ne
                if not found:
                    for c in chunked[i][j]:
                        if not type(c) is nltk.Tree:
                            if c[0].lower() in words:
                                for (l, ne) in nes:
                                    if l == label:
                                        found = ne
                                        break
                        if found:
                            break
                    if firstFound == None:
                        for (l, ne) in nes:
                            if l == label:
                                firstFound = ne
                                break
                if found:
                    break
            if found:
                break
        return (found, firstFound)
                      

    def searchHeader(self, header, key, label):
        if key in header:
            chunked = self.nerTagger.process("\n".join(header[key]))
            for i in range(len(chunked)):
                for j in range(len(chunked[i])):
                    nes = self.nerTagger.getNesFromTree(self.nerTagger.neTagSentence(chunked[i][j]))
                    for (l, ne) in nes:
                        if l == label:
                            return ne
        return None

    def headerToDict(self, header):
        headers = {}
        current = None
        headers[current] = []
        splitHeader = header.split("\n")
        for s in splitHeader:
            found = re.match("([A-Za-z]+):(.+)", s)
            if found == None:
                headers[current].append(s)
            else:
                current = (found.group(1))
                headers[current] = [found.group(2)]
        return headers

    def extractHeaderAbstract(self, raw):
        splitRaw = raw.split("\n") 
        for i in range(len(splitRaw)):
            if splitRaw[i].startswith("Abstract:"):
                return ("\n".join(splitRaw[:i]), "\n".join(splitRaw[(i+1):] if i+1 < len(splitRaw) else []))
        return (raw, None)
         
        for p in corpus.paras():
            para = ""
            if p != []:
                for s in p:
                    if s != []:
                        paras = ("<sentence>" + "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in s]).strip()) + "</sentence>"
            para += "<paragraph>" + paras[1:] + "</paragraph>"
            para += "\n"

class OntologyMaker():

    def __init__(self, seminarDict):
        categories = {}
        toBeClassified = set()
        for sid in list(seminarDict):
            seminar = seminarDict[sid]                          
            if seminar == None:
                continue
            _type = seminar._type
            if not _type == None:
                if "." in _type:
                    splitType = _type.split(".")
                    subcategory = splitType[-1]
                    category = splitType[-2]
                    if not category.lower() in categories.keys():
                        categories[category.lower()] = {}
                        categories[category.lower()][subcategory.lower()] = [sid]
                    elif not subcategory.lower() in categories[category.lower()].keys():
                        categories[category.lower()][subcategory.lower()] = [sid]
                    else:
                        getArray = categories[category.lower()][subcategory.lower()]
                        getArray.append(sid)
                        categories[category.lower()][subcategory.lower()] = getArray
                else:
                    _type = _type.lower()
                    _type = _type.replace("seminar","")
                    _type = _type.strip(" ")
                    splitType = _type.split(" ")
                    tokens = nltk.word_tokenize(_type)
                    found = False
                    subcategory = None
                    category = splitType[0]
                    if len(splitType) > 1:
                        subcategory = splitType[1] 
                    else:
                        subcategory = "other"
                    for k in categories.keys():
                        for v in categories[k].keys():
                            for t in splitType:
                                if t == v:
                                    category = k
                                    subcategory = v
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not category.lower() in categories.keys():
                        categories[category.lower()] = {}
                        categories[category.lower()][subcategory.lower()] = [sid]
                    elif not subcategory.lower() in categories[category.lower()].keys():
                        categories[category.lower()][subcategory.lower()] = [sid]
                    else:
                        getArray = categories[category.lower()][subcategory.lower()]
                        getArray.append(sid)
                        categories[category.lower()][subcategory.lower()] = getArray
                        
        self.categories = categories
                           
    def search(self, query):
        d = self.categories
        for k in d.keys():
            if k == query:
                printDict(d[k])
                continue
            for v in d[k].keys():
                if v == query:
                    printDict(d[k][v])

    def printDict(self, d):
        for k in d.keys():
            if type(d[k]) is dict:
                for k2 in d[k].keys():
                    print(d[k][k2])
            else:
                print(d[k])
             
    def printOntology(self):
        for k in self.categories.keys():
            print(k)
            for v in self.categories[k].keys():
                print("\t"+v)
                for f in self.self.categories[k][v]:
                    print("\t\t"+f)


class SeminarTester():

    def locationTest(tagger, files):
        count = 0
        for f in files:
            print(f)
            s = Seminar.makeSeminarFromFile("./test_tagged/"+f)
            myS = tagger.tag("./test_untagged/",f)
            if s == None or myS == None:
                continue
            if str(s.location).replace(" ", "") == str(myS.location).replace(" ", ""):
                count += 1
            else:
                print(s.location, myS.location)
        return (100 * count / len(files))

    def timeTest(tagger, files):
        count = 0
        ignored = 0
        for f in files:
            print(f)
            s = Seminar.makeSeminarFromFile("./test_tagged/"+f)
            myS = tagger.tag("./test_untagged/", f)
            if s == None or myS == None:
                ignored += 1
                continue
            if Seminar.compareLooseTime(s.startTime, myS.startTime) and Seminar.compareLooseTime(s.endTime, myS.endTime):
                count += 1
            else:
                print(myS.startTime, myS.endTime, s.startTime, s.endTime)
        return (100 * count / (len(files) - ignored))

    def speakerTest(tagger, files):
        count = 0
        ignored = 0
        for f in files:
            print(f)
            s = Seminar.makeSeminarFromFile("./test_tagged/"+f)
            myS = tagger.tag("./test_untagged/", f)
            if s == None or myS == None:
                ignored += 1
                continue
            if s.speaker == myS.speaker:
                count += 1
            else:
                print(s.speaker, myS.speaker)
        return (100 * count / (len(files) - ignored))

                
nerTrainPath = "./wsj_training/"
files = [f for f in listdir(path) if isfile(join(path, f))]
nerTrainFiles = [f for f in listdir(nerTrainPath) if isfile(join(nerTrainPath, f))][:1400]
#files = ["311.txt","312.txt","313.txt","349.txt","314.txt","315.txt","316.txt","317.txt","318.txt","319.txt", "400.txt"]
#files = ["436.txt", "432.txt"]
#files = ["403.txt"]
nerTagger = NERTagger()
nerTagger.train(nerTrainPath, nerTrainFiles)
tagger = SeminarTagger(nerTagger)
count = 0
seminarDict = {}

print(SeminarTester.locationTest(tagger, files))

#for f in files:
#    print(f)
#    myS = tagger.tag(path, f)
#    seminarDict[f] = myS
#
#ontology = OntologyMaker(seminarDict)
#ontology.printOntology()

