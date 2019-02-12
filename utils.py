from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from random import shuffle
from tqdm import trange
import numpy as np
import json
import h5py
import csv
import re
import nltk
import six
from nltk.corpus import mac_morpho
import requests
import gzip
import gensim
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re


def querysuggestion_sample(preds, temperature, interactive=False, top_n=3):
    '''
    Samples predicted probabilities of the next character to allow
    for the network to show "creativity."
    '''

    preds = np.asarray(preds).astype('float64')

    if temperature is None or temperature == 0.0:
        return np.argmax(preds)

    preds = np.log(preds + K.epsilon()) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    if not interactive:
        index = np.argmax(probas)

        # prevent function from being able to choose 0 (placeholder)
        # choose 2nd best index from preds
        if index == 0:
            index = np.argsort(preds)[-2]
    else:
        # return list of top N chars/words
        # descending order, based on probability
        index = (-preds).argsort()[:top_n]

    return index









#######################################################################Import Stopwords From Stopword_txt_file#######################################################

def Stopword_Retrieve():
    from_filename="C:/Users\MD REZAUL ISLAM/Desktop/Query Suggestion_rnn_final/querysuggestion/STOP_WORD.txt"
    from_file_read = open(from_filename, 'r', encoding="UTF-8")
    contents_read = from_file_read.read()


    stop_words = ""
    for i in contents_read.split("\n"):
        stop_words+=i+" "
    stop_words=re.sub('[^ ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]',"",stop_words)
    stop_words_split=stop_words.split()
    ulist = []
    [ulist.append(x) for x in stop_words_split if x not in ulist]
    return ulist



############################################################################Removing_Query_Word######################################################################

def Removing_Word(not_similar,adding_final_document):

    
    split_final_document=adding_final_document.split() 

    split_not_similar=not_similar.split()
    stop=''
    length_split_final_document=len(split_final_document)
    length_split_not_similar=len(split_not_similar)

    for i in range(0,length_split_final_document):
      for j in range(0,length_split_not_similar):
          if(split_not_similar[j] in split_final_document[i]):
              stop+=split_final_document[i]
              stop+=' '
              break
    
    not_similar_final_document_=""
    not_similar_final_document=' '.join([i for i in split_final_document if i not in stop.split()])
    return not_similar_final_document










############################################################################Removing_Similar_Word######################################################################




def unique_list(text):
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    search_key=""
    Retrieved_Stopword=Stopword_Retrieve()
    for q in ulist:
          #for r in q.split():
             #if r not in Retrieved_Stopword: 
                search_key+=q+" "

    
    search_key=search_key.rstrip()
    return search_key






####################################################Retrieving Similar Word for a word Using Word2Vec############################################################


def Removing_Stemming_word(Final):
    import difflib
    Filter=Final.copy()
    Last=[]
    for i in Final:
        if i in Filter:
         Last.append(i)
         Filter.remove(i)
         for j in Filter:
            value=difflib.SequenceMatcher(None, i, j).ratio()
            if value>0.51 :
                    Filter.remove(j);

    return Last

def Retrieve_Similar_Word(word,model):
    import difflib
    word_vectors = model.wv
    not_similar_unique=""
    FINAL_OUTPUT=""
    if word in word_vectors.vocab:
        word2vec=model.wv.most_similar(word, topn=50)
        iword=''.join(str(e) for e in word2vec)                   #string covert
        clean_word2vec=''
        for i in iword:
              line = re.sub('[^ \nঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃ৅েৈ৉োৌ্ৎৗ৘ড়ঢ়য়০১২৩৪৫৬৭৮৯]', '', i[0])
              clean_word2vec+=line


        not_similar=''.join([i for i in word])+' '      
        not_similar=Removing_Word(word,clean_word2vec)
        not_similar_unique=unique_list(not_similar.split())


     # Removing spelling error word
        score = difflib.SequenceMatcher(None, word,not_similar_unique.split()).ratio()
        
        FINAL_OUTPUT=''.join([i for i in word])+' '
        #print(not_similar_unique)
 
        for i in not_similar_unique.split():
            value=difflib.SequenceMatcher(None, word, i).ratio()
            #print(i,'-->>>',value)
            if(value<0.51):
                FINAL_OUTPUT+=i+" "
                
    FINAL_OUTPUT=Removing_Stemming_word(FINAL_OUTPUT.split())
    FINAL_OUTPUT=' '.join(str(e) for e in FINAL_OUTPUT)
    return FINAL_OUTPUT






model1 = Word2Vec.load('D:/200D_model_cbow/word2vec_model.model')
print(model1)




def querysuggestion_generate(model, vocab,
                        indices_char, temperature=0.5,
                        maxlen=40, meta_token='<s>',
                        word_level=False,
                        single_text=False,
                        max_gen_length=300,
                        interactive=False,
                        top_n=3,
                        prefix=None,
                        synthesize=False,
                        stop_tokens=[' ', '\n']):
    '''
    Generates and returns a single text.
    '''

    collapse_char = ' ' if word_level else ''
    end = False

    # If generating word level, must add spaces around each punctuation.
    # https://stackoverflow.com/a/3645946/9314418
    if word_level and prefix:
        punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'
        prefix = re.sub('([{}])'.format(punct), r' \1 ', prefix)
        prefix_t = [x.lower() for x in prefix.split()]

    if not word_level and prefix:
        prefix_t = list(prefix)

    if single_text:
        text = prefix_t if prefix else ['']
        max_gen_length += maxlen
    else:
        text = [meta_token] + prefix_t if prefix else [meta_token]

    next_char = ''

    if not isinstance(temperature, list):
        temperature = [temperature]

    if len(model.inputs) > 1:
        model = Model(inputs=model.inputs[0], outputs=model.outputs[1])

    while not end and len(text) < max_gen_length:
        encoded_text = querysuggestion_encode_sequence(text[-maxlen:],
                                                  vocab, maxlen)
        next_temperature = temperature[(len(text) - 1) % len(temperature)]

        if not interactive:
            # auto-generate text without user intervention
            next_index = querysuggestion_sample(
                model.predict(encoded_text, batch_size=1)[0],
                next_temperature)
            next_char = indices_char[next_index]
            text += [next_char]
            if next_char == meta_token or len(text) >= max_gen_length:
                end = True
            gen_break = (next_char in stop_tokens or word_level or
                         len(stop_tokens) == 0)
            if synthesize and gen_break:
                break
        else:
            # ask user what the next char/word should be
            options_index = querysuggestion_sample(
                model.predict(encoded_text, batch_size=1)[0],
                next_temperature,
                interactive=interactive,
                top_n=top_n
            )
            options = [indices_char[idx] for idx in options_index]
            print('Controls:\n\ts: stop.\tx: backspace.\to: write your own query.')
            print('\nSuggestions:')

            """

            for i, option in enumerate(options, 1):
                print('\t{}: {}'.format(i, option))

            """
            i=1
            Display_Words=[]
            for option in options:
                
                Similar_words=Retrieve_Similar_Word(option,model1)
                Similar_words_5=Similar_words.split()[:5]
                for word in Similar_words_5:
                  print('\t{}: {}'.format(i, word))
                  Display_Words.append(word)
                  i=i+1

            print('\nProgress: {}'.format(collapse_char.join(text)[3:]))
            print('\nYour choice?')
            user_input = input('> ')

#############################################################################################
            
            #model = Word2Vec.load('D:/200D_model_cbow/word2vec_model.model')
            #print(model)
            try:
                user_input = int(user_input)
                print(len(Display_Words))
                next_char = Display_Words[user_input-1]
                print(next_char)
                
                #print("\n",Retrieve_Similar_Word(str(next_char),model))
                text += [next_char]
            except ValueError:
                if user_input == 's':
                    next_char = '<s>'
                    text += [next_char]
                elif user_input == 'o':
                    other = input('> ')
                    text += [other]
                elif user_input == 'x':
                    try:
                        del text[-1]
                    except IndexError:
                        pass
                else:
                    print('That\'s not an option!')

    # if single text, ignore sequences generated w/ padding
    # if not single text, remove the <s> meta_tokens
    if single_text:
        text = text[maxlen:]
    else:
        text = text[1:]
        if meta_token in text:
            text.remove(meta_token)

    text_joined = collapse_char.join(text)

    # If word level, remove spaces around punctuation for cleanliness.
    if word_level:
        #     left_punct = "!%),.:;?@]_}\\n\\t'"
        #     right_punct = "$([_\\n\\t'"
        punct = '\\n\\t'
        text_joined = re.sub(" ([{}]) ".format(punct), r'\1', text_joined)
        #     text_joined = re.sub(" ([{}])".format(
        #       left_punct), r'\1', text_joined)
        #     text_joined = re.sub("([{}]) ".format(
        #       right_punct), r'\1', text_joined)

    return text_joined, end


def querysuggestion_encode_sequence(text, vocab, maxlen):
    '''
    Encodes a text into the corresponding encoding for prediction with
    the model.
    '''

    encoded = np.array([vocab.get(x, 0) for x in text])
    return sequence.pad_sequences([encoded], maxlen=maxlen)


def querysuggestion_texts_from_file(file_path, header=True,
                               delim='\n', is_csv=False):
    '''
    Retrieves texts from a newline-delimited file and returns as a list.
    '''

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        if is_csv:
            texts = []
            reader = csv.reader(f)
            for row in reader:
                texts.append(row[0])
        else:
            texts = [line.rstrip(delim) for line in f]

    return texts


def querysuggestion_texts_from_file_context(file_path, header=True):
    '''
    Retrieves texts+context from a two-column CSV.
    '''

    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        texts = []
        context_labels = []
        reader = csv.reader(f)
        for row in reader:
            texts.append(row[0])
            context_labels.append(row[1])

    return (texts, context_labels)


def querysuggestion_encode_cat(chars, vocab):
    '''
    One-hot encodes values at given chars efficiently by preallocating
    a zeros matrix.
    '''

    a = np.float32(np.zeros((len(chars), len(vocab) + 1)))
    rows, cols = zip(*[(i, vocab.get(char, 0))
                       for i, char in enumerate(chars)])
    a[rows, cols] = 1
    return a


def synthesize(textgens, n=1, return_as_list=False, prefix='',
               temperature=[0.5, 0.2, 0.2], max_gen_length=300,
               progress=True, stop_tokens=[' ', '\n']):
    """Synthesizes texts using an ensemble of input models.
    """

    gen_texts = []
    iterable = trange(n) if progress and n > 1 else range(n)
    for _ in iterable:
        shuffle(textgens)
        gen_text = prefix
        end = False
        textgen_i = 0
        while not end:
            textgen = textgens[textgen_i % len(textgens)]
            gen_text, end = querysuggestion_generate(textgen.model,
                                                textgen.vocab,
                                                textgen.indices_char,
                                                temperature,
                                                textgen.config['max_length'],
                                                textgen.META_TOKEN,
                                                textgen.config['word_level'],
                                                textgen.config.get(
                                                    'single_text', False),
                                                max_gen_length,
                                                prefix=gen_text,
                                                synthesize=True,
                                                stop_tokens=stop_tokens)
            textgen_i += 1
        if not return_as_list:
            print("{}\n".format(gen_text))
        gen_texts.append(gen_text)
    if return_as_list:
        return gen_texts


def synthesize_to_file(textgens, destination_path, **kwargs):
    texts = synthesize(textgens, return_as_list=True, **kwargs)
    with open(destination_path, 'w') as f:
        for text in texts:
            f.write("{}\n".format(text))


class generate_after_epoch(Callback):
    def __init__(self, querysuggestion, gen_epochs, max_gen_length):
        self.querysuggestion = querysuggestion
        self.gen_epochs = gen_epochs
        self.max_gen_length = max_gen_length

    def on_epoch_end(self, epoch, logs={}):
        if self.gen_epochs > 0 and (epoch+1) % self.gen_epochs == 0:
            self.querysuggestion.generate_samples(
                max_gen_length=self.max_gen_length)


class save_model_weights(Callback):
    def __init__(self, querysuggestion, num_epochs, save_epochs):
        self.querysuggestion = querysuggestion
        self.weights_name = querysuggestion.config['name']
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs

    def on_epoch_end(self, epoch, logs={}):
        if len(self.querysuggestion.model.inputs) > 1:
            self.querysuggestion.model = Model(inputs=self.model.input[0],
                                          outputs=self.model.output[1])
        if self.save_epochs > 0 and (epoch+1) % self.save_epochs == 0 and self.num_epochs != (epoch+1):
            print("Saving Model Weights — Epoch #{}".format(epoch+1))
            self.querysuggestion.model.save_weights(
                "{}_weights_epoch_{}.hdf5".format(self.weights_name, epoch+1))
        else:
            self.querysuggestion.model.save_weights(
                "{}_weights.hdf5".format(self.weights_name))
