import nltk
class Test():
    def calculate(mine, test):
        count = 0
        if len(test) == 0:
            return None
        for neT in test:
            neMIndex = 0
            while neMIndex < len(mine):
                if neT[0] == mine[neMIndex][0] and neT[1] == mine[neMIndex][1]:
                    count += 1
                    mine.remove(mine[neMIndex])
                    break
                neMIndex += 1
        return count / (len(test))

    def looseCalculate(mine, test):
        count = 0
        if len(test) == 0:
            return None
        for neT in test:
            neMIndex = 0
            while neMIndex < len(mine):
                if neT[0] == mine[neMIndex][0]:
                    if mine[neMIndex][1] in neT[1] or neT[1] in mine[neMIndex][1]:
                        count += 1
                        mine.remove(mine[neMIndex])
                        break
                neMIndex += 1
        return count / (len(test))
