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
from sklearn.pipeline import Pipeline
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
    """Takes in a list of text, run doc2vec on it, outputs vectors"""

    def __init__(self, d2v_dim, context_win):
        self.d2v_dim = d2v_dim
        self.context_win = context_win

    def transform(self, X, y=None):
        """The workhorse of this feature extractor"""
        return get_d2v(X, self.d2v_dim, self.context_win)

    def fit(self, X, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


def get_d2v(txts, d2v_dim, context_win):
    sents = LabeledLineSentence(txts)
    model = Doc2Vec(sents, min_count=1, window=context_win, size=d2v_dim, sample=1e-4, negative=5, workers=7)
    for epoch in range(50):
        logger.info('Epoch %d' % epoch)
        model.train(sents)

    feat_array = numpy.zeros((len(txts), d2v_dim))
    for i in range(len(txts)):
        prefix = 'SENT_' + str(i)
        feat_array[i] = model.docvecs[prefix].tolist()
    return feat_array


with open('vword/trans.txt', 'r') as fin:
    txts = fin.readlines()
    # d2v = get_d2v(txts, d2v_dim=50, context_win=10)

    # use transformer directly
    # d2v_tf = d2v_transformer(50, 10)
    # d2v = d2v_tf.transform(txts)

    # using transformer in a pipeline.
    ppl = Pipeline([('d2v', d2v_transformer(50, 10))])
    d2v = ppl.transform(txts)
    print(d2v)

