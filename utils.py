""" Utils functions for final project
Alina G and Jesse E
Text
 """ 
import numpy as np 
import librosa
import speech_recognition as sr
import json

def load_wavs(pathname: str, files:list, get_labels=True) -> list: 
    """ takes a list of paths leading to .wav files and reads them. Also gets 
        the labels if needed from the titles. Pathname is the path the files
        are in, assuming there is a folder. 
    """
    values = []
    for file in files: 
        if file.endswith('.wav'):
            sig, rate = librosa.load(pathname + '/' + file, sr=None)
            sig_rate = (sig.tolist(), rate)
            code_lst = file.split('-') # splits the file_name up into fragments 
            emotion_label = int(code_lst[2]) 
            values.append((sig_rate, emotion_label)) 
    return values 

def get_audio_text(pathname:str, files:list) -> list: 
    """ using pathname and list of files, attempts to convert audio to text
    *** ASSIGNS 0 FOR AN UNRECOGNIZED AUDIO FILE***
    """
    texts = []
    # intialize the recognizer...
    r = sr.Recognizer()
    for file in files: 
        if file.endswith('.wav'): 
            code_lst = file.split('-') # splits the file_name up into fragments 
            emotion_label = int(code_lst[2]) 
            with sr.AudioFile(pathname + '/' + file) as source:
                audio = r.record(source)
                try: 
                    s = r.recognize_google(audio)
                except: 
                    s = 0 # in case our recognizer fails, we will just assign 0...
                texts.append((s, emotion_label))
    return texts

def seperate_tups(lst:list) -> tuple: 
    """ takes in a list of tuple parings and returns a tuple of 2 lists """ 
    x, y = [], []
    for pair in lst: 
        x.append(pair[0])
        y.append(pair[1])
    return x, y

def write_files(dct:dict, path: str)-> None: 
    """ writes files into json format """
    with open(path, 'w') as file: 
        json.dump(dct, file)
    print(f'Successfully dumped data into {path}!')



class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    self.n_ = n_gram
    self.probabilities_ = {}
    self.n_grams = []
    self.vocab = []
    self.tokens = []
    self.limit_ = 10 # limiter for sentence generation

  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """  
    # gets the probabilities for each ngram occuring and stores in dct
    for i in range(len(tokens)):
      toks_ = tokens[i] 
      count_w = tokens.count(toks_)
      conditional = count_w == 1 and (toks_ != SENTENCE_BEGIN or toks_ != SENTENCE_END)

      # checks if UNK
      if conditional: 
        self.tokens.append(UNK)
      else: 
        self.tokens.append(toks_)

    # updates vocab and n_grams  
    self.vocab = list(set(self.tokens))
    self.n_grams = create_ngrams(self.tokens, self.n_)
    
    # assigns probability in dct 
    for toks in self.vocab: 
      count_w = self.tokens.count(toks)
      for tup in self.n_grams: 
        if toks.lower() == tup[self.n_-1].lower(): 
          
          count_ngram = self.n_grams.count(tup)
          self.probabilities_[tup] = count_ngram / count_w
  

  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
      
    Returns:
      float: the probability value of the given tokens for this model
    """
    # STUDENTS IMPLEMENT
    probs = []
    for i in range(len(sentence_tokens)): 
      if self.n_ > 1:
        # currently only supports bigram
        try: 
          wi, wi_1 = sentence_tokens[i], sentence_tokens[i+1]
          if wi not in self.tokens: 
            wi = UNK
          if wi_1 not in self.tokens: 
            wi_1 = UNK
          tup = (wi, wi_1)
        except: 
          # in case of index out of range
          break
        num = self.n_grams.count(tup) + 1
        den = self.tokens.count(wi) + len(self.vocab)
        probs.append(num / den)
      else:  
        # for unigram models
        wi = tuple([sentence_tokens[i]])
        count = self.n_grams.count(wi)

        # checks to see if value is not in training, and assigns it <UNK>
        num = 0
        if count == 0: 
          num = self.n_grams.count((UNK,)) + 1
        else: 
          num = count + 1

        # get the probability
        den = len(self.n_grams) + len(self.vocab)
        probs.append(num / den)
    
    # returns the product of all scores in probs
    return math.prod(probs)

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # start sentence, i represents current pos in text
    generated_text = [SENTENCE_BEGIN]
    i = 0
    # keep generating while the end of sentence is False 
    while generated_text[i] != SENTENCE_END: 
      poss_next = []
      weights = []
      for tup in self.n_grams: 
        if generated_text[i] == tup[0] and self.n_ > 1: 
          # save possible next words along with their probabilities for weights
          poss_next.append(tup[1])
          weights.append(self.probabilities_[tup])
        elif self.n_ == 1: 
          # if unigram, all toks have same weight (exclude <s>)
          poss_next = [_ for _ in self.tokens if _ != SENTENCE_BEGIN]
          weights = [self.probabilities_[key] for key in self.n_grams if SENTENCE_BEGIN not in key]
          # no need to loop for all ngrams, so we just break
          break
      # calculate next word w/ weights
      next_word = random.choices(poss_next, weights=weights)
      # ensures no repeated start sentences 
      while generated_text[i] == SENTENCE_BEGIN and next_word[0] == SENTENCE_BEGIN: 
        next_word = random.choices(poss_next, weights=weights)
      generated_text += next_word
      i += 1
      # added a limiter to prevent the program from derailing and never hitting the end
      if i > self.limit_: 
        generated_text[i] = SENTENCE_END

    # return ' '.join(generated_text)
    return generated_text

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]

  
