######################################################################
#  CliNER - model.py                                                 #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: Define the model for clinical concept extraction.        #
######################################################################

from __future__ import with_statement

from sklearn.feature_extraction  import DictVectorizer
import os
import random
import math
import io

from time        import localtime, strftime
from collections import defaultdict

from machine_learning   import crf
from notes.documents    import labels as tag2id, id2tag
from tools              import flatten, save_list_structure, reconstruct_list

# Stores the verbosity

import numpy as np
from feature_extraction.features import extract_features


class ClinerModel:

    def log(self, out, model_file=None):
        '''
        ClinerModel::log()
        Log training information of model.
        @param out.         Either a filename or file channel to output the log string.
        @param model_file.  A path to optionally identify where the model was saved.
        @return None
        '''
        if not self._log:
            log = self.__log_str(model_file)
        else:
            log = self._log

        # depending on whether it is already opened as a channel
        if isinstance(out,file):
            print >>out, log
        else:
            with open(out, 'a') as f:
                print >>f, log


    def __log_str(self, model_file=None):
        '''
        ClinerModel::__log_str()
        Build a string of information about training for the model's log file.
        @param model_file.  A path to optionally identify where the model was saved.
        @return  A string of the model's training information
        '''
        assert self._is_trained, 'ClinerModel not trained'
        with io.StringIO() as f:
            f.write(u'\n')
            f.write(unicode('-'*40))
            f.write(u'\n\n')
            if model_file:
                f.write(unicode('model    : %s\n' % os.path.abspath(model_file)))
                f.write(u'\n')

            if self._use_lstm:
                f.write(u'modeltype: LSTM\n')
            else:
                f.write(u'modeltype: CRF\n')

            if 'hyperparams' in self._score:
                for name,value in self._score['hyperparams'].items():
                    f.write(u'\t%-10s: %s\n' % (name,value))
            f.write(u'\n')
            
            print_str(f, 'features', self._features)
            f.write(u'\n')

            f.write(u'\n')
            f.write(unicode('training began: %s\n' % self._time_train_begin))
            f.write(unicode('training ended: %s\n' % self._time_train_end))
            f.write(u'\n')

            f.write(u'scores\n')
            print_vec(f, 'train precision', self._score['train']['precision'])
            print_vec(f, 'train recall   ', self._score['train']['recall'   ])
            print_vec(f, 'train f1       ', self._score['train']['f1'       ])
            f.write(self._score['train']['conf'])

            if 'dev' in self._score:
                print_vec(f, u'dev precision   ', self._score['dev']['precision'])
                print_vec(f, u'dev recall      ', self._score['dev']['recall'   ])
                print_vec(f, u'dev f1          ', self._score['dev']['f1'       ])
                f.write(self._score['dev']['conf'])

            if 'history' in self._score:
                for label,vec in self._score['history'].items():
                    print_vec(f, '%-16s'%label, vec)
                f.write(u'\n')

            if self._training_files:
                f.write(u'\n')
                f.write(u'Training Files\n')
                if len(self._training_files) < 200:
                    print_files(f, self._training_files)
                else:
                    f.write(unicode('\t%d files\n'%len(self._training_files)))
                f.write(u'\n')
                
            f.write(u'-'*40)
            f.write(u'\n\n')

            # get output as full string
            contents = f.getvalue()
        return contents


    def __init__(self, use_lstm):

        """
        ClinerModel::__init__()

        Instantiate a ClinerModel object.

        @param use_lstm. Bool indicating whether to train a CRF or LSTM.
        """
        self._use_lstm       = use_lstm
        self._is_trained     = False
        self._clf            = None
        self._vocab          = None
        self._training_files = None
        self._log            = None
        self._text_feats     = None



    def train(self, notes):
        """
        ClinerModel::train()

        Purpose: Train a Machine Learning model on annotated data

        @param notes. A list of Note objects (containing text and annotations)
        @return       None
        """

        # Extract formatted data
        tokenized_sentences = flatten([n.getTokenizedSentences() for n in notes])
        labels              = flatten([n.getTokenLabels() for n in notes])
       
        self.train_fit(tokenized_sentences, labels, dev_split=0.1)

        self._training_files = [ n.getName() for n in notes ]


    def train_fit(self,tok_sents, tags, val_sents=None, val_tags=None, dev_split=None):
        """
        ClinerModel::train_fit()

        Purpose: Train clinical concept extraction model using annotated data.

        @param tok_sents.   A list of sentences, where each sentence is tokenized into words.
        @param tags.        Parallel to 'tokenized_sents', 7-way labels for concept spans.
        @param val_sents.   Validation data. Same format as tokenized_sents
        @param val_tags.    Validation data. Same format as iob_nested_labels
        @param dev_split    A real number from 0 to 1
        """
        # metadata
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # train classifier
        voc, clf, dev_score, enabled_features = generic_train('all', 
                                                              tok_sents             ,
                                                              tags                  ,
                                                              self._use_lstm        ,
                                                              val_sents=val_sents   , 
                                                              val_labels=val_tags   ,
                                                              dev_split=dev_split   )
        
        self._is_trained = True
        self._vocab = voc
        self._clf   = clf
        self._score = dev_score
        self._features = enabled_features

        # metadata
        self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())


    def predict_classes_from_document(self, document):
        """
        ClinerModel::predict_classes_from_documents()

        Predict concept annotations for a given document

        @param note. A Document object (containing text and annotations)
        @return      List of predictions
        """
        # Extract formatted data
        tokenized_sents  = document.getTokenizedSentences()

        return self.predict_classes(tokenized_sents)


    def predict_classes(self, tokenized_sents):
        """
        ClinerModel::predict_classes()

        Predict concept annotations for unlabeled, tokenized sentences

        @param tokenized_sents. A list of sentences, where each sentence is tokenized
                                  into words
        @return                  List of predictions
        """
        # Predict labels for prose
        num_pred = generic_predict('all'                     ,
                                    tokenized_sents          ,
                                    vocab    = self._vocab   ,
                                    clf      = self._clf     ,
                                    use_lstm = self._use_lstm)
        
        iob_pred = [ [id2tag[p] for p in seq] for seq in num_pred ]

        return iob_pred

    ############################################################################
    ###               Lowest-level (interfaces to ML modules)                ###
    ############################################################################

def generic_train(p_or_n, tokenized_sents, iob_nested_labels, use_lstm, val_sents=None, val_labels=None, dev_split=None):

    '''
    generic_train()

    Train a model that works for both prose and nonprose

    @param p_or_n.             A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents.    A list of sentences, where each sentence is tokenized
                                 into words
    @param iob_nested_labels.  Parallel to `tokenized_sents`, 7-way labels for 
                                 concept spans
    @param use_lstm            Bool indicating whether to train CRF or LSTM.
    @param val_sents.          Validation data. Same format as tokenized_sents
    @param val_labels.         Validation data. Same format as iob_nested_labels
    @param dev_split.          A real number from 0 to 1
    '''

    # Must have data to train on:
    if len(tokenized_sents) == 0:
        raise Exception('Training must have %s training examples' % p_or_n)

    # if you should split the data into train/dev yourself
    #if (not val_sents) and (dev_split > 0.0) and (len(tokenized_sents)>1000):
    if (not val_sents) and (dev_split > 0.0) and (len(tokenized_sents)>10):

        p = int(dev_split*100)
        print '\tCreating %d/%d train/dev split' % (100-p,p)

        perm = range(len(tokenized_sents))
        random.shuffle(perm)

        tokenized_sents   = [   tokenized_sents[i] for i in perm ]
        iob_nested_labels = [ iob_nested_labels[i] for i in perm ]

        ind = int(dev_split*len(tokenized_sents))

        val_sents   = tokenized_sents[:ind ]
        train_sents = tokenized_sents[ ind:]

        val_labels   = iob_nested_labels[:ind ]
        train_labels = iob_nested_labels[ ind:]

        tokenized_sents   = train_sents
        iob_nested_labels = train_labels


    print '\tvectorizing words', p_or_n

    #tokenized_sents   = train_sents[ :2]
    #iob_nested_labels = train_labels[:2]

    # count word frequencies to determine OOV
    freq = defaultdict(int)
    for sent in tokenized_sents:
        for w in sent:
            freq[w] += 1

    # determine OOV based on % of vocab or minimum word freq threshold
    oov = set()
    '''
    if len(freq) < 100:
        lo = len(freq)/20
        oov = set([ w for w,f in sorted(freq.items(), key=lambda t:t[1]) ][:lo])
    else:
        #lo = 2
        #oov = set([ w for w,f in freq.items() if (f <= lo) ])
        oov = set()
    '''

    '''
    val = None
    for w,f in sorted(freq.items(), key=lambda t:t[1]):
        if val != f:
            val = f
            print
        print '%8d  %s' % (f,w)
    exit()
    '''

    if use_lstm:
        ########
        # LSTM
        ########

        # build vocabulary of words
        vocab = {}
        for sent in tokenized_sents:
            for w in sent:
                if (w not in vocab) and (w not in oov):
                    vocab[w] = len(vocab) + 1
        vocab['oov'] = len(vocab) + 1

        # vectorize tokenized sentences
        X_seq_ids = []
        for sent in tokenized_sents:
            id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
            X_seq_ids.append(id_seq)

        # vectorize IOB labels
        Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in iob_nested_labels ]

        # if there is specified validation data, then vectorize it
        if val_sents:
            # vectorize validation X
            val_X = []
            for sent in val_sents:
                id_seq = [ (vocab[w] if w in vocab else vocab['oov']) for w in sent ]
                val_X.append(id_seq)
            # vectorize validation Y
            val_Y = [ [tag2id[y] for y in y_seq] for y_seq in val_labels ]

    else:
        ########
        # CRF
        ########

        # vectorize tokenized sentences
        '''
        def make_feature(ind):
            return {(ind,i):1 for i in range(10)}
        text_features = []
            fseq = [make_feature(vocab[w] if w in vocab else vocab['oov']) for w in sent]
            text_features.append(fseq)
        '''
        text_features = extract_features(tokenized_sents)
        # type(text_features): <type 'list'>
        # text_features is a list of list of dicts
        
        # Collect list of feature types
        enabled_features = set()
        for sf in text_features:
            for wf in sf:
                for (feature_type,instance),value in wf.items():
                    if feature_type.startswith('prev'):
                        feature_type = 'PREV*'
                    if feature_type.startswith('next'):
                        feature_type = 'NEXT*'
                    enabled_features.add(feature_type)
        enabled_features = sorted(enabled_features)

        
        # Vectorize features
        vocab = DictVectorizer()
        flat_X_feats = vocab.fit_transform( flatten(text_features) )
        X_feats = reconstruct_list(flat_X_feats, save_list_structure(text_features))

        # vectorize IOB labels
        Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in iob_nested_labels ]

        assert len(X_feats) == len(Y_labels)
        for i in range(len(X_feats)):
            assert X_feats[i].shape[0] == len(Y_labels[i])

        # if there is specified validation data, then vectorize it
        if val_sents:
            # vectorize validation X
            val_text_features = extract_features(val_sents)
            flat_val_X_feats = vocab.transform( flatten(val_text_features) )
            val_X = reconstruct_list(flat_val_X_feats,
                                     save_list_structure(val_text_features))
            # vectorize validation Y
            val_Y = [ [tag2id[y] for y in y_seq] for y_seq in val_labels ]


    print '\ttraining classifiers', p_or_n

    if use_lstm:
        # train using lstm
        clf, dev_score  = keras_ml.train(X_seq_ids, Y_labels, tag2id, len(vocab),
                                         val_X_ids=val_X, val_Y_ids=val_Y)
    else:
        # train using crf
        clf, dev_score  = crf.train(X_feats, Y_labels, val_X=val_X, val_Y=val_Y)

    return vocab, clf, dev_score, enabled_features



def generic_predict(p_or_n, tokenized_sents, vocab, clf, use_lstm):
    '''
    generic_predict()

    Train a model that works for both prose and nonprose

    @param p_or_n.          A string that indicates "prose", "nonprose", or "all"
    @param tokenized_sents. A list of sentences, where each sentence is tokenized
                              into words
    @param vocab.           A dictionary mapping word tokens to numeric indices.
    @param clf.             An encoding of the trained keras model.
    @param use_lstm.        Bool indicating whether clf is a CRF or LSTM.
    '''

    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict ' + p_or_n
        return []

    print '\tvectorizing words ' + p_or_n

    if use_lstm:
        # vectorize tokenized sentences
        X = []
        for sent in tokenized_sents:
            id_seq = []
            for w in sent:
                if w in vocab:
                    id_seq.append(vocab[w])
                else:
                    id_seq.append(vocab['oov'])
            X.append(id_seq)
    else:
        # vectorize validation X
        text_features = extract_features(tokenized_sents)
        flat_X_feats = vocab.transform( flatten(text_features) )
        X = reconstruct_list(flat_X_feats, save_list_structure(text_features))

    print '\tpredicting  labels ' + p_or_n

    # Predict labels
    if use_lstm:
        predictions = keras_ml.predict(clf, X)
    else:
        predictions =   crf.predict(clf, X)

    # Format labels from output
    return predictions



def print_files(f, file_names):
    '''
    print_files()

    Pretty formatting for listing the training files in a 
    log.

    @param f.           An open file stream to write to.
    @param file_names.  A list of filename strings.
    '''
    COLUMNS = 4
    file_names = sorted(file_names)
    start = 0
    for row in range(len(file_names)/COLUMNS + 1):
        f.write(u'\t\t')
        for featname in file_names[start:start+COLUMNS]:
            f.write(unicode('%-15s' % featname))
        f.write(u'\n')
        start += COLUMNS



def print_vec(f, label, vec):
    '''
    print_vec()

    Pretty formatting for displaying a vector of numbers in a log.

    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param vec.    A numpy array of the numbers to display.
    '''
    COLUMNS = 7
    start = 0
    f.write(unicode('\t%-10s: ' % label))
    if type(vec) != type([]):
        vec = vec.tolist()
    for row in range(len(vec)/COLUMNS):
        for featname in vec[start:start+COLUMNS]:
            f.write(unicode('%7.3f' % featname))
        f.write(u'\n')
        start += COLUMNS
        
def print_str(f, label, names):

    '''
    print_str()
    Pretty formatting for displaying a list of strings in a log
    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param names.  A list of strings.
    '''
    COLUMNS = 4
    start = 0
    for row in range(int(math.ceil(float(len(names))/COLUMNS))):
        if row == 0:
            f.write(unicode('\t%-10s: ' % label))
        else:
            f.write(unicode('\t%-10s  ' % ''))

        for featname in names[start:start+COLUMNS]:
            f.write(unicode('%-16s ' % featname))
            
        f.write(u'\n')
        start += COLUMNS
