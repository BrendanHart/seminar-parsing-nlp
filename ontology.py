import nltk, re

class OntologyMaker():
    
    def makeOntology(seminars):

        seminarDict = {}

        for s in seminars:
            
