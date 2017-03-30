# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys
# import cPickle as pickle
import pprint

# sklearn
from sklearn.base import TransformerMixin
from sklearn.svm import LinearSVC

# Update train_d2v.py to be able to
# - adding a clf
# - using pipeline to connect d2v and clf together
# - hyper-parameter tweak

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

class LabeledLineSentence(object):
    """now uses a list of text (txts) as input rather than source"""
    def __init__(self, txts):
        self.txts = txts

    def __iter__(self):
        for item_no, line in enumerate(self.txts):
            yield TaggedDocument(utils.to_unicode(line).split(), ['SENT' + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for item_no, line in enumerate(self.txts):
            self.sentences.append(TaggedDocument(
                        utils.to_unicode(line).split(), ['SENT' + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class d2v_transformer(TransformerMixin):
    """Takes in dataframe, extracts road name column, outputs average word length"""

    def __init__(self, d2v_dim):
        self.d2v_dim = d2v_dim

    def d2v_txts(self, X):
        sentences = LabeledLineSentence(X)
        model = Doc2Vec(sentences, min_count=1, window=10, size=self.d2v_dim, sample=1e-4, negative=5, workers=7)
        # model.build_vocab(sentences.to_array())
        for epoch in range(50):
            logger.info('Epoch %d' % epoch)
            model.train(sentences.sentences_perm())

        feat_array = numpy.zeros((len(X), self.d2v_dim))
        for i in range(len(X)):
            prefix = 'SENT_' + str(i)
            feat_array[i] = model.docvecs[prefix].tolist()
        return feat_array

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return self.d2v_txts(X)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self



with open('vword/trans.txt', 'r') as fin:
    txts = fin.readlines()
    sents = LabeledLineSentence(txts)
    model = Doc2Vec(sents, min_count=1, window=10, size=50, sample=1e-4, negative=5, workers=7)
    for epoch in range(50):
        logger.info('Epoch %d' % epoch)
        model.train(sents)

    feat_array = numpy.zeros((len(txts), 50))
    for i in range(len(txts)):
        prefix = 'SENT_' + str(i)
        feat_array[i] = model.docvecs[prefix].tolist()

print(feat_array)

