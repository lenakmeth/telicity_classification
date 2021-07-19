import io
import json
import os
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, CamembertTokenizer, FlaubertTokenizer
from configure import parse_args
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

args = parse_args()

def read_sents(path, marker):
    """ Read the .tsv files with the annotated sentences. 
        File format: sent_id, sentence, verb, verb_idx, label"""

    def open_file(file):
        sentences = []
        labels = []
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                sentences.append([l[-4], l[-3], int(l[-2])])
                labels.append(int(l[-1]))
                
            return sentences,labels
        
    train_sentences, train_labels = open_file(path + '/' + marker + '_train.tsv')    
    val_sentences, val_labels = open_file(path +  '/' + marker + '_val.tsv')
    test_sentences, test_labels = open_file(path +  '/' + marker + '_test.tsv')

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels

            

def tokenize_and_pad(sentences):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    
    if (args.transformer_model).split("-")[0] == 'bert':
        tok = BertTokenizer.from_pretrained(args.transformer_model)
    elif (args.transformer_model).split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.transformer_model)
    elif (args.transformer_model).split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(args.transformer_model )
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = XLNetTokenizer.from_pretrained(args.transformer_model )
    elif 'flaubert' in args.transformer_model:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = FlaubertTokenizer.from_pretrained(args.transformer_model )
    elif 'camembert' in args.transformer_model:
        tok = CamembertTokenizer.from_pretrained(args.transformer_model)
        
    for sent in sentences:
        encoded_dict = tok.encode_plus(
                            sent[0],                      
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,      # Pad & truncate all sentences.
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True, # Construct attn. masks.
                            # return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(
            encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        # Add segment ids, add 1 for verb idx
        segment_id = [0] * 128
        
        verb_idx = find_real_verb_idx(sent, encoded_dict['input_ids'], tok)
       
        if verb_idx: # if False, the verb is not in the first 128 tokens
            for idx in verb_idx:
                segment_id[idx] = 1
            
        segment_ids.append(segment_id)     

    return input_ids, attention_masks, segment_ids


def find_real_verb_idx(sentence_list, encoded_sentence, tokenizer):
    
    encoded_verb = tokenizer.encode(sentence_list[1])

    try:
        if len(encoded_verb) == 3: # verb as is + [CLS] + [SEP]
            verb_idx = [encoded_sentence.index(encoded_verb[1])]
        else:
            decoded_verb = tokenizer.convert_ids_to_tokens(encoded_verb)
            decoded_sent = tokenizer.convert_ids_to_tokens(encoded_sentence)
            verb_segment = [seg for seg in decoded_verb 
                            if not any(seg.startswith(x) for x in ['[', '<'])]
            verb_idx = [decoded_sent.index(seg) for seg in verb_segment]
            
        return verb_idx
    
    except ValueError:
#         print(sentence_list)
        # the sequence is way too long and the verb is chopped!
        return False


def flat_accuracy(labels, preds):
    """ Function to calculate the accuracy of our predictions vs labels. """
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

    
def decode_result(encoded_sequence):

    if (args.transformer_model).split("-")[0] == 'bert':
        tok = BertTokenizer.from_pretrained(args.transformer_model )
    elif (args.transformer_model).split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.transformer_model )
    elif (args.transformer_model).split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(args.transformer_model )
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        tok = XLNetTokenizer.from_pretrained(args.transformer_model )
    elif 'camembert' in args.transformer_model:
        tok = CamembertTokenizer.from_pretrained(args.transformer_model )
    elif 'flaubert' in args.transformer_model:
        tok = FlaubertTokenizer.from_pretrained(args.transformer_model )
    
    # decode + remove special tokens
    tokens_to_remove = ['[PAD]', '[SEP]', '[CLS]', '<pad>', '<sep>', '<s>', '</s>']
    decoded_sequence = [w.replace('Ġ', '').replace('▁', '').replace('</w>', '')
                        for w in list(tok.convert_ids_to_tokens(encoded_sequence))
                        if not w.strip() in tokens_to_remove]
    
    return ' '.join(decoded_sequence)


def split_train_val_test(X, y, 
                         frac_train=0.8, frac_val=0.1, frac_test=0.1,
                         random_state=2018):

    # Split original dataframe into train and temp dataframes.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=(1.0 - frac_train),
                                                        random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    stratify=y_temp,
                                                    test_size=relative_frac_test,
                                                    random_state=random_state)

    assert len(X) == len(X_train) + len(X_val) + len(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_attn(words, attentions, layer, heads):
    """Plots attention maps for the given example and attention heads."""
    width = 3
    example_sep = 3
    word_height = 1
    pad = 0.1

    for ei, head in enumerate(heads):
        yoffset = 1
        xoffset = ei * width * example_sep

        attn = attentions[layer][head]
        attn = np.array(attn)
#         print(attn.shape)
        attn /= attn.sum(axis=-1, keepdims=True)
        n_words = len(words)

        for position, word in enumerate(words):
            plt.text(xoffset + 0, yoffset - position * word_height, word,
                   ha="right", va="center")
            plt.text(xoffset + width, yoffset - position * word_height, word,
                   ha="left", va="center")
        for i in range(1, n_words):
            for j in range(1, n_words):
                alpha_size = float(attn[i, j].item())

                plt.plot([xoffset + pad, xoffset + width - pad],
                 [yoffset - word_height * i, yoffset - word_height * j],
                 color="blue", linewidth=1, alpha=alpha_size)


def get_features_per_layer(features, layer_idx): 

    features_per_layer = {x:[] for x in range(len(features['all_hidden_states']))}
    for sent_idx in features_per_layer:
        temp = extract_features(features, sent_idx)
        temp_layer = []
        for token_idx in temp:
            layer_features = [x[layer_idx] for x in temp.values()] 
            layer_features = np.concatenate(layer_features)
            temp_layer.append(layer_features)

        features_per_layer[sent_idx] = np.concatenate(temp_layer)
        
    return features_per_layer
