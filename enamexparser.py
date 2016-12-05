import re
import nltk
class EnamexParser():

    def __init__(self, path, files):
        self.fileDict = {}
        count = 0
        for f in files:
            print("Processing training file: " + str(count+1))
            text = open(path+f, 'r').read()
            self.fileDict[f] = self.parse(text)
            count += 1
        print("Done parsing.")

    def parse(self, text):
    
        entities = ["PERSON", "ORGANIZATION", "LOCATION"] 
        punct = "."

        # Get the named entities then the text without them
        ne = re.findall(r'TYPE="(.+?)">(.+?)<', text)
        splitOnNe = re.split('<ENAMEX TYPE=".+?">.+?</ENAMEX>', text)

        # Tag each token with it's tagged named entity type
        entityTagged = []
        for i in range(len(splitOnNe)):
            if i < len(ne):
                entityTagged += [(x, "O") for x in nltk.word_tokenize(splitOnNe[i])] + [(x, ne[i][0]) for x in nltk.word_tokenize(ne[i][1])]
            else:
                entityTagged += [(x, "O") for x in nltk.word_tokenize(splitOnNe[i])]
            
        
        ### Fix up Corp . being separate.
        i = 0
        while (i < len(entityTagged)):
            (x, y) = entityTagged[i]
            if x in punct and y in entities and i-1 >= 0 and entityTagged[i-1][1]==y:
                entityTagged[i-1] = (entityTagged[i-1][0]+x, y)
                entityTagged.pop(i)
                i -= 1
            i += 1

        ### Add pos tags
        wordsFromEntityTagged = nltk.pos_tag([x for (x, y) in entityTagged])
        for i in range(len(entityTagged)):
            entityTagged[i] = (entityTagged[i][0], entityTagged[i][1], wordsFromEntityTagged[i][1])

        return entityTagged

    def get(self, fileID):
        return self.fileDict[fileID]

    def nesFromParsed(parsed):
        
        listOfNes = []
        shouldAdd = False
        current = ""
        currentTag = None
        for t in parsed:
            if t[1] == "O":
                if shouldAdd:
                    listOfNes.append((currentTag, current[1:]))
                shouldAdd = False
                current = ""
                currentTag = None
            else:
                shouldAdd = True
                current += " " + t[0]
                currentTag = t[1]
        if shouldAdd:
            listOfNes.append((currentTag, current[1:]))

        return listOfNes

    def neTreesFromParsed(parsed):
        classes = ["ORGANIZATION", "LOCATION", "PERSON"]
        trees = []
        for i in range(len(parsed)):
            if parsed[i][1] in classes:
                wordList = []
                after = None
                for j in range(i, len(parsed)):
                    if(parsed[j][1] == parsed[i][1]):
                        wordList.append((parsed[j][0], parsed[j][2]))
                    else:
                        if j+1 < len(parsed):
                            after = (parsed[j+1][0], parsed[j+1][2])
                        break
                tree = nltk.Tree(parsed[i][1], [wordList, after])
                trees.append(tree)
        return trees

