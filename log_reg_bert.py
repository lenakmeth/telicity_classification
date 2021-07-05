#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, XLNetForSequenceClassification, CamembertForSequenceClassification, FlaubertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, CamembertTokenizer, FlaubertTokenizer
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


# In[2]:


def tokenize_and_pad(transformer_model, sentences):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    
    if transformer_model.split("-")[0] == 'bert':
        tok = BertTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'xlnet':
        tok = XLNetTokenizer.from_pretrained(transformer_model)
    elif 'camembert' in transformer_model:
        tok = CamembertTokenizer.from_pretrained(transformer_model)
    elif 'flaubert' in transformer_model:
        tok = FlaubertTokenizer.from_pretrained(transformer_model)

    for sent in sentences:
        sentence = sent[0]

        # encode_plus is a prebuilt function that will make input_ids, 
        # add padding/truncate, add special tokens, + make attention masks 
        encoded_dict = tok.encode_plus(
                        sentence,                      
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,      # Pad & truncate all sentences.
                        padding = 'max_length',
                        truncation = True,
                        return_attention_mask = True, # Construct attn. masks.
                        # return_tensors = 'pt',     # Return pytorch tensors.
                   )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # Add segment ids, add 1 for verb idx
        segment_id = encoded_dict['token_type_ids']
        segment_id[sent[2]] = 1
        segment_ids.append(segment_id)

        attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks, segment_ids


def decode_result(transformer_model, encoded_sequence):

    if transformer_model.split("-")[0] == 'bert':
        tok = BertTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(transformer_model)
    elif transformer_model.split("-")[0] == 'xlnet':
        tok = XLNetTokenizer.from_pretrained(transformer_model)
    elif 'camembert' in transformer_model:
        tok = CamembertTokenizer.from_pretrained(transformer_model)
    elif 'flaubert' in transformer_model:
        tok = FlaubertTokenizer.from_pretrained(transformer_model)
    
    # decode + remove special tokens
    tokens_to_remove = ['[PAD]', '<pad>', '<s>', '</s>']
    decoded_sequence = [w.replace('Ġ', '').replace('▁', '').replace('</w>', '')
                        for w in list(tok.convert_ids_to_tokens(encoded_sequence))
                        if not w.strip() in tokens_to_remove]
    
    return decoded_sequence


# In[3]:


# load finetuned model

tokenizer_model = 'bert-base-uncased'
verb_segment_ids = 'no'
model_save_path = 'checkpoints/friedrich_captions_data/telicity/'
batch_size = 64


if tokenizer_model.split("-")[0] == 'bert':
    model = BertForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)
elif tokenizer_model.split("-")[0] == 'roberta':
    model = RobertaForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)
elif tokenizer_model.split("-")[0] == 'albert':
    model = AlbertForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)
elif tokenizer_model.split("-")[0] == 'xlnet':
    model = XLNetForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)
elif 'camembert' in tokenizer_model:
    model = CamembertForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)
elif 'flaubert' in tokenizer_model:
    model = FlaubertForSequenceClassification.from_pretrained(model_save_path + tokenizer_model + '_' + verb_segment_ids)


# In[5]:


# open train set

train_sentences = []
train_labels = []
    
with open('data/friedrich_captions_data/telicity_train.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip().split('\t')
        train_sentences.append([l[-4], l[-3], int(l[-2])])
        train_labels.append(int(l[-1]))

use_segment_ids = False
train_inputs, train_masks, train_segments = tokenize_and_pad(tokenizer_model, train_sentences)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_segments = torch.tensor(train_segments)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# In[6]:


# open val set

val_sentences = []
val_labels = []
    
with open('data/friedrich_captions_data/telicity_val.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip().split('\t')
        val_sentences.append([l[-4], l[-3], int(l[-2])])
        val_labels.append(int(l[-1]))

use_segment_ids = False
val_inputs, val_masks, val_segments = tokenize_and_pad(tokenizer_model, val_sentences)

val_inputs = torch.tensor(val_inputs)
val_labels = torch.tensor(val_labels)
val_masks = torch.tensor(val_masks)
val_segments = torch.tensor(val_segments)

# open test set

test_sentences = []
test_labels = []
    
with open('data/friedrich_captions_data/telicity_test.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip().split('\t')
        test_sentences.append([l[-4], l[-3], int(l[-2])])
        test_labels.append(int(l[-1]))

use_segment_ids = False
test_inputs, test_masks, test_segments = tokenize_and_pad(tokenizer_model, test_sentences)

test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)
test_segments = torch.tensor(test_segments)


def get_outputs(data_dataloader):
    model.eval()

    all_inputs = []
    sent_attentions = []
    sentences = []
    predicted_labels = []
    prob_prediction = []
    all_hidden_states = []

    features = {'all_inputs': [],  
                'all_labels': [],
                # 'all_dec_sentences': [],
                'all_lengths': [],
                # 'all_attentions': [], 
                # 'all_pred_labels': [], 
                # 'all_prob_predictions': [], 
                'all_hidden_states': []}

    for n, batch in enumerate(data_dataloader):
        print(n, len(data_dataloader))
        # Add batch to GPU
        batch = tuple(t for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():        

            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            output_hidden_states=True)

            logits = outputs[0]
            all_inputs = b_input_ids.to('cpu').numpy().tolist()
            features['all_inputs'] += all_inputs
            logits = logits.detach().cpu().numpy()
            
            label_ids = b_labels.to('cpu').numpy()
            features['all_labels'] += label_ids.tolist()
            
            hidden_states = outputs[1]
#             attentions = outputs[-1]

             # Add predictions and probabilities
#             log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)
            # features['all_pred_labels'] += np.argmax(logits, axis=1).flatten().tolist()
#             features['all_prob_predictions'] += log_probs.tolist()

            # Decode sentence, as model sees it 
#             sentence = decode_result(tokenizer_model, inputs)
#             features['all_dec_sentences'].append(sentence)
            for sent_inputs in all_inputs:
            	sentence = decode_result(tokenizer_model, sent_inputs)
            	features['all_lengths'].append(len(sentence))

        #Add hidden states

        hidden_states_per_sentence = {sent_idx:[] for sent_idx in range(batch_size)}

        for layer in hidden_states:
            for sent_idx in range(batch_size):
                try:
                    hidden_states_per_sentence[sent_idx] += layer[sent_idx, :, :]
                except IndexError:
                    print(layer.shape, sent_idx)

        features['all_hidden_states'] += list(hidden_states_per_sentence.values())
        
    return features


# In[ ]:


train_features = get_outputs(train_dataloader)
# val_features = get_outputs(val_inputs, val_labels)
# test_features = get_outputs(test_inputs, test_labels)


# In[ ]:


def extract_features(features, sent_idx):

    hidden_states = features['all_hidden_states'][sent_idx]
    seq_length = features['all_lengths'][sent_idx]

    features_per_word = {x:[] for x in range(seq_length)}

    for layer_out in hidden_states:
        # layer = torch.squeeze(layer_out) # remove batch dim = 1
        layer = np.array(layer)[:seq_length, :] # trim out padding in seq
        for n in range(seq_length): 
            features_per_word[n].append(layer[n]) # features of every word per layer
            
    return features_per_word


# In[ ]:


# how this looks:
# each sent_idx has a dictionary of:
# keys = token position
# values = list of 13 layers (pool + bert layers), each layer has embedding of (768,)

# extract now embeddings for each token, per layer
layer_idx = -1

def get_features_per_layer(features, layer_idx): 

    features_per_layer = {x:[] for x in range(len(features['all_hidden_states']))}
    for sent_idx in features_per_layer:
        temp = extract_features(features, sent_idx)
        temp_layer = []
        for token_idx in temp:
    #         print(temp.keys())
            layer_features = [x[layer_idx] for x in temp.values()] 
            layer_features = np.concatenate(layer_features)
            temp_layer.append(layer_features)

        features_per_layer[sent_idx] = np.concatenate(temp_layer)
        
    return features_per_layer


# In[ ]:


# make the sets for the log regression model and pad as necessary

# train
train_features_per_layer = get_features_per_layer(train_features, layer_idx)
max_len = max([len(x) for x in train_features_per_layer.values()])
padded_X = [np.pad(x, (0, max_len-len(x)), 'constant') for x in train_features_per_layer.values()]

X_train = np.asarray(padded_X)
y_train = np.asarray(train_features['all_labels'])

# val
val_features_per_layer = get_features_per_layer(val_features, layer_idx)
max_len = max([len(x) for x in val_features_per_layer.values()])
padded_X = [np.pad(x, (0, max_len-len(x)), 'constant') for x in val_features_per_layer.values()]

X_val = np.asarray(padded_X)
y_val = np.asarray(val_features['all_labels'])

# test
test_features_per_layer = get_features_per_layer(test_features, layer_idx)
max_len = max([len(x) for x in test_features_per_layer.values()])
padded_X = [np.pad(x, (0, max_len-len(x)), 'constant') for x in test_features_per_layer.values()]

X_test = np.asarray(padded_X)
y_test = np.asarray(test_features['all_labels'])


# In[ ]:


logit = LogisticRegression(C=1e-2, random_state=17, solver='lbfgs', 
                           multi_class='multinomial', max_iter=100,
                          n_jobs=4)


logit.fit(X_train, y_train)

logit_val_pred = logit.predict_proba(X_val)
result_val = log_loss(y_val, logit_val_pred)
print(result_val)

logit_test_pred = logit.predict_proba(X_test)
result_test = log_loss(y_test, logit_test_pred)
print(result_test)

logger = open('log_reg_results.txt', 'w+')
logger.write(tokenizer_model + ' layer: ' + str(layer_idx))
logger.write('\nval: ' + str(result_val))
logger.write('\ntest: ' + str(result_test) + '\n')
