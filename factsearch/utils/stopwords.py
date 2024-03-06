class StopWordList:
    def __init__(self, fname="data/stopwords/czech.txt"):
        with open(fname) as f:
            self.stopwords = set([l.strip().lower() for l in f.readlines()])

    def is_stopword(self, word):
        return word.strip().lower() in self.stopwords

         


