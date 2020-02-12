import transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM
 
import torch
import distance
import copy
import pickle
import json
import math
import numpy as np
import re

def load_voc_set(dictfile):
    return set(line.strip() for line in open(dictfile))

def load_training_set_voc(pklfile):
    return pickle.load(open(pklfile,'rb'))

def load_voc_fromlist(voclist):
    return set(term.strip() for term in voclist)

def get_closest_word(ww,wset):
    bestd=100
    bestw=''

    for v in wset:
       #print(v+' ' +str(distance.levenshtein(v, self.original_sentence[mei])))
       if distance.levenshtein(v.lower(), ww.lower())<bestd:
           bestd = distance.levenshtein(v.lower(), ww.lower())
           bestw = v
            
    return bestw

def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

class bert_lm:
    
    def __init__(self, modelpath):
       self.modelpath = modelpath
       self.tokenizer = BertTokenizer.from_pretrained(modelpath)
       self.model = BertForMaskedLM.from_pretrained(modelpath)
       self.voc = load_voc_set(modelpath+'/vocab.txt')
       print(len(self.voc))
       self.model.eval()
       print ('Bert LM model initialized')
        
    def expandvoc(self, voclist):
        self.voc = self.voc.union(load_voc_fromlist(voclist))

    def resetvoc(self):
        self.voc = load_voc_set(self.modelpath+'/vocab.txt')
        
    def tokenize(self, input):
       self.tokenized_text = self.tokenizer.tokenize(input)
       print (self.tokenized_text)
       self.tokenize_text = self.tokenized_text.insert(len(self.tokenized_text), '[MASK]')
       print (self.tokenized_text)
       self.masked_index = len(self.tokenized_text)-1
    
    def get_score(self,intext):
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(intext)])
        predictions=self.model(tensor_input)[0]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data 
        return math.exp(loss)    
    
    def tokenize_for_oov(self, input):
       self.masked_index = -1 
#        self.tokenized_text = self.tokenizer.tokenize(input)
#        print (self.tokenized_text)
       newsent = input.split(' ')
        
       for i in range(len(newsent)):
           if newsent[i].lower() not in self.voc:
              print(newsent[i] + ' is not found')
              self.maked_word = newsent[i]
              newsent[i]='[MASK]'
              self.masked_index = i
           
       self.tokenized_text = newsent
       return ' '.join(newsent)
 
    def tokenize_for_oov_v2(self, input, rules=None):
       self.masked_index = [] 
       newsent = input.split(' ')
       self.original_sentence = copy.deepcopy(newsent) 
       for i in range(len(newsent)):
            
           ifignore = False 
           if rules is not None:
              
              for r in rules:
                if bool(re.match(r, newsent[i])):
                    ifignore = True
                    break
                
           if ((newsent[i].lower() not in self.voc) and (not ifignore)):
              print(newsent[i] + ' is not found')
              #self.maked_word = newsent[i]
            
              newsent[i]='[MASK]'
              # adding the masked word index 
              self.masked_index.append(i)
              #print('Put '+ self.original_sentence[i] + ' as MASK')  
                
       self.tokenized_text = newsent
    
       return ' '.join(newsent)
 
    def tokenize_for_oov_v3(self, input):
       self.masked_index = [] 
       newsent = input.split(' ')
       self.original_sentence = copy.deepcopy(newsent) 
       for i in range(len(newsent)):
           if ((newsent[i].lower() not in self.voc) & (re.match(pattern, newsent[i]))):
              print(newsent[i] + ' is not found')
              # adding the masked word index 
              self.masked_index.append(i)
              #print('Put '+ self.original_sentence[i] + ' as MASK')  
                
       self.tokenized_text = newsent
    
       return ' '.join(newsent)
    
    def predict_oov(self, n):
       if self.masked_index !=-1:
           print ('Masked sentence '+ ' '.join(self.tokenized_text) + ' and the masked index is ' + str(self.masked_index)) 
           indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenized_text)
           segments_ids = [0]*len(self.tokenized_text)
           tokens_tensor = torch.tensor([indexed_tokens])
           segments_tensors = torch.tensor([segments_ids])
 
           #print (indexed_tokens) 
           with torch.no_grad():
               outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
               predictions_1 = outputs[0]
           #predicted_index = torch.argmax(predictions_1[0, masked_index]).item()
           sorted_index = torch.argsort(predictions_1[0, self.masked_index], descending=True).tolist()
           returnstr = []
 
           for loop in range(n):
               predicted_token = self.tokenizer.convert_ids_to_tokens([sorted_index[loop]])[0]
               returnstr.append(predicted_token)
               #returnstr.append(' '.join(self.tokenized_text).replace('[MASK]', predicted_token))
               #print(predicted_token)
 
           bestd=100
           bestw=''
 
           for v in returnstr:
               if distance.levenshtein(v, self.maked_word)<bestd:
                   bestd = distance.levenshtein(v, self.maked_word)
                   bestw = v
                   bestsent = ' '.join(self.tokenized_text)
               #print(v+' ' +str(distance.levenshtein(v, self.maked_word)))
           bestsent=self.tokenized_text
           bestsent[self.masked_index]=bestw
           print ('Recovered sentence ' + ' '.join(bestsent)) 
           return returnstr, bestw, ' '.join(bestsent)       
       else:
           return ' ', ' ', ' '.join(self.tokenized_text)
 
    def predict_oov_v2(self, n):
       if len(self.masked_index)!=0:
           print ('Masked sentence '+ ' '.join(self.tokenized_text) + ' and the masked index is ' + str(self.masked_index)) 
           indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenized_text)
           segments_ids = [0]*len(self.tokenized_text)
           tokens_tensor = torch.tensor([indexed_tokens])
           segments_tensors = torch.tensor([segments_ids])
 
           #print (indexed_tokens) 
           with torch.no_grad():
               outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
               predictions_1 = outputs[0]
           #predicted_index = torch.argmax(predictions_1[0, masked_index]).item()
        
           bestsent=self.tokenized_text
            
           # looping the masked index
           for mei in self.masked_index:
                
               sorted_index = torch.argsort(predictions_1[0, mei], descending=True).tolist()
               sorted_probability = [predictions_1[0, mei][i] for i in sorted_index]
            
               returnstr = []
 
               for loop in range(n):
                   predicted_token = self.tokenizer.convert_ids_to_tokens([sorted_index[loop]])[0]
                   returnstr.append(predicted_token)
                   #returnstr.append(' '.join(self.tokenized_text).replace('[MASK]', predicted_token))
                   #print(predicted_token)
 
               bestd=100
               bestw=''
 
               for v in returnstr:
                   #print(v+' ' +str(distance.levenshtein(v, self.original_sentence[mei])))
                   if distance.levenshtein(v.lower(), self.original_sentence[mei].lower())<bestd:
                       bestd = distance.levenshtein(v.lower(), self.original_sentence[mei].lower())
                       bestw = v
                       
               
               bestsent[mei]=bestw
                
           print ('Recovered sentence ' + ' '.join(bestsent)) 
           return ' '.join(bestsent),self.get_score(bestsent)      
        
       else:
           return ' '.join(self.tokenized_text), self.get_score(self.tokenized_text)
        
        
    def predict_oov_v3(self, n):
       if len(self.masked_index)!=0:
           print ('Masked sentence '+ ' '.join(self.tokenized_text) + ' and the masked index is ' + str(self.masked_index)) 
           indexed_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenized_text)
           segments_ids = [0]*len(self.tokenized_text)
           tokens_tensor = torch.tensor([indexed_tokens])
           segments_tensors = torch.tensor([segments_ids])
 
           #print (indexed_tokens) 
           with torch.no_grad():
               outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
               predictions_1 = outputs[0]
           #predicted_index = torch.argmax(predictions_1[0, masked_index]).item()
        
           bestsent=self.tokenized_text
            
           # looping the masked index
           # beam search     
            
           best_p_instore=[]
           best_i_instore=[]
        
           icount = 0
            
           for mei in self.masked_index:
               
                
               # obtaining index of word id in decending order at position mei
               sorted_index = torch.argsort(predictions_1[0, mei], descending=True).tolist()
               #sorted_index = [[i] for i in sorted_index]
            
               
               sorted_probability = [predictions_1[0, mei][i] for i in sorted_index]
           
        
               if ((len(best_p_instore)==0) or (icount==0)):
                   comb_prob_buf= sorted_probability
                   comb_prob_index = sorted_index 
               else:
                   comb_prob_buf = np.fromiter([best_p_instore[i]*sorted_probability[j] for i in range(len(best_p_instore)) for j in range(len(sorted_probability))], float)
                   comb_prob_index = [flatten([best_i_instore[i], sorted_index[j]]) for i in range(len(best_index_combo)) for j in range(len(sorted_probability))]
            
            
            
               # sort the array of mulitplication  
               sorted_comb_index = np.flip(np.argsort(comb_prob_buf))
               sorted_comb_value = np.sort(comb_prob_buf)
               sorted_comb_value = sorted_comb_value[::-1] 

                
               # cut the array size by n, the beamsize
               best_comb_index = sorted_comb_index[0:n]
               best_comb_value = sorted_comb_value[0:n]
               best_index_combo = [comb_prob_index[i] for i in best_comb_index]
                
               best_p_instore = best_comb_value 
               best_i_instore = best_index_combo

#                best_index_combo = sorted_comb_index
#                best_p_instore = sorted_comb_value
#                best_i_instore = sorted_comb_index

               icount+=1 
           print('best i instore is ' + str(best_i_instore))
        
        
           #print('best i instore word is ' + str([self.tokenizer.convert_ids_to_tokens([i])[0] for i in best_i_instore]))
           bestd  = 10000
           # when the iteration is done, check the top n 
        
           for e in best_index_combo:
              sum_score = 0   
            
              if len(self.masked_index)>1:
                  icombo = 0
                    
                  for ii in self.masked_index:
                      predicted_token = self.tokenizer.convert_ids_to_tokens([e[icombo]])[0]
                      sum_score = sum_score + distance.levenshtein(predicted_token.lower(), self.original_sentence[self.masked_index[icombo]].lower())
                      icombo+=1
              else:
                  predicted_token = self.tokenizer.convert_ids_to_tokens([e])[0]
                  sum_score = distance.levenshtein(predicted_token.lower(), self.original_sentence[self.masked_index[0]].lower())
                    
              if sum_score< bestd:
                  besti = e
                  bestd = sum_score  
               
           # obtain the best comb
           if len(self.masked_index)==1:
               bestsent[self.masked_index[0]]=self.tokenizer.convert_ids_to_tokens([besti])[0]
           else:
               for ii in range(len(self.masked_index)):
                   bestsent[self.masked_index[ii]]=self.tokenizer.convert_ids_to_tokens([besti[ii]])[0]
                
           print ('Recovered sentence ' + ' '.join(bestsent)) 
           return ' '.join(bestsent),self.get_score(bestsent)       
       else:
           return ' '.join(self.tokenized_text), self.get_score(self.tokenized_text)     
        
global bertlm

bertlm = bert_lm('/mnt/bert_model_serve_sharepoint/')
        
def test(intext, beamsize=10,  voclist=None, rule=None):
    global bertlm
    
    if voclist is not None:
        voclist = voclist.split('|')
        bertlm.expandvoc(voclist)
    else:
        bertlm.resetvoc()
    
    if rule is not None:
        rule = rule.split('|')
        masked_text = bertlm.tokenize_for_oov_v2(intext, rule) 
    else:
        masked_text = bertlm.tokenize_for_oov_v2(intext)
        
    if beamsize>100:
         b, c =bertlm.predict_oov_v2(beamsize) 
    else:
         b, c =bertlm.predict_oov_v3(beamsize) 
          
    return b, c


# sent, score = test('Does it require cutomer siganture for IRA signup', 1000, 'asds|sdaa', '^[A-Z ]+$')
# print(sent)
# print (score)
