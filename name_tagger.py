from nltk.tag import SequentialBackoffTagger
from nltk.corpus import names


class NamesTagger(SequentialBackoffTagger):
    def __init__(self, *args, **kwargs):
        SequentialBackoffTagger.__init__(self, *args, **kwargs)
        self.name_set = set([n.lower() for n in names.words()])

    def choose_tag(self, tokens, index, history) :
        word = tokens[index]

        if word.lower() in self.name_set:
            return 'NNP'
        else :
            return None

            
