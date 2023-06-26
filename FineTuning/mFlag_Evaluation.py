import time
start_time = time.time()

import transformers
import torch
import bert_score
#import bleu
import nltk
#nltk.download('punkt')
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(transformers.__version__)
print(torch.__version__)
from mFLAG.model import MultiFigurativeGeneration
from mFLAG.tokenization_mflag import MFlagTokenizerFast
tokenizer = MFlagTokenizerFast.from_pretrained('laihuiyuan/mFLAG')
model = MultiFigurativeGeneration.from_pretrained('laihuiyuan/mFLAG')

bert_scorer = bert_score.BERTScorer(lang='en', rescale_with_baseline=True)

def evaluate(link, amount):
  model.eval()
  idioms_df = pd.read_csv(link,encoding = "ISO-8859-1")
  fig = list(idioms_df["Figurative sentence"].sample(n=amount, random_state=42))
  lit = list(idioms_df["Literal meaning"].sample(n=amount, random_state=42))
  #fig = list(idioms_df["Figurative sentence"][:amount])
  #lit = list(idioms_df["Literal meaning"][:amount])
  type=idioms_df["Type"][0]
  hyps=[]
  refs=lit
  counter=0
  for sentence in fig:
    counter+=1

    #insert model-----------------------
    inp_ids = tokenizer.encode(
        "<"+type.lower()+">"+sentence.strip(),return_tensors="pt")
    fig_ids = tokenizer.encode("<literal>", add_special_tokens=False, return_tensors="pt")
    outs = model.generate(input_ids=inp_ids[:, 1:], fig_ids=fig_ids, forced_bos_token_id=fig_ids.item(), num_beams=5,
                          max_length=60, )
    result = tokenizer.decode(outs[0, 2:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # special tokens: <literal>, <hyperbole>, <idiom>, <sarcasm>, <metaphor>, or <simile>
    #insert model-----------------------

    if (counter%5 == 0):
      print("Generating at "+str(counter)+" times")
    hyps.append(result)
  return hyps, refs

def score(hyps, refs, type):
  #hyps = [nltk.word_tokenize(line.strip(), language='english') for line in hyps]
  #refs = [nltk.word_tokenize(line.strip(), language='english') for line in refs]

  if type=='bert':
    scores = []
    for l1,l2 in zip(hyps, refs):
      _, _, bert_score = bert_scorer.score([l1], [l2])
      scores.append(round(bert_score[0].tolist(),4))

  elif type=='bleu':
    scores = sentence_bleu(refs, hyps)
  print('The average score is {}'.format(np.mean(scores)))

hyps, refs = evaluate("C:\HQC\AI4GoodLab\LiteraLingo\Attempt\Datasets\Hyperbole.csv",100)
print(score(hyps, refs,'bert'))
print(score(hyps, refs,'bleu'))

common_elements = list(set(hyps).intersection(refs))
print(common_elements)










# Print the running time
end_time = time.time()
running_time = end_time - start_time
print("Running time:", running_time, "seconds")