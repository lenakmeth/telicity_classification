import io
import json
import os
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer
from configure import parse_args
import time
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

args = parse_args()

def read_sents(marker):
    """ Read the .tsv files with the annotated sentences. 
        File format: sent_id, sentence, verb, verb_idx, label"""

    def open_file(file):
        sentences = []
        labels = []
        
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                l = line.strip().split('\t')
                sentences.append([l[1], l[2], int(l[3])])
                labels.append(int(l[4]))
                
            return sentences,labels
        
    train_sentences, train_labels = open_file('data/' + marker + '_train.tsv')
    val_sentences, val_labels = open_file('data/' + marker + '_val.tsv')
    test_sentences, test_labels = open_file('data/' + marker + '_test.tsv')

    return train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels


def tokenize_and_pad(sentences, lowercase=False):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    
    
    if (args.transformer_model).split("-")[0] == 'bert':
        # Tokenize all of the sentences and map the tokens to their word IDs.
        if 'uncased' in args.transformer_model.split("-"):
            tok = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        else:
            tok = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
        
        
        for sent in sentences:
            
            # lowercase if needed by the model
            if lowercase:
                sentence = [w.lower() for w in sent[0]]
            else:
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
    
    # ======== RoBERTa ========
    elif (args.transformer_model).split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
        
        for sent in sentences:
            # encode_plus is a prebuilt function that will make input_ids, 
            # add padding/truncate, add special tokens, + make attention masks 
            encoded_dict = tok.encode_plus(
                            sent[0],                      
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,      # Pad & truncate all sentences.
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True, # Construct attn. masks.
                            # return_tensors = 'pt',     # Return pytorch tensors.
                       )
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            
            # This attention mask simply differentiates padding from non-padding.
            attention_masks.append(encoded_dict['attention_mask'])
            
        # RoBERTa doesn't use segment ids!  
        segment_ids = None
    
    # ======== AlBERT ========
    elif (args.transformer_model).split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
        
        for sent in sentences:
            # encode_plus is a prebuilt function that will make input_ids, 
            # add padding/truncate, add special tokens, + make attention masks 
            encoded_dict = tok.encode_plus(
                            sent[0],                      
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

            # This attention mask simply differentiates padding from non-padding.
            attention_masks.append(encoded_dict['attention_mask'])

    # ======== XLNET ======== 
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = XLNetTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
        
        for sent in sentences:

            # encode_plus is a prebuilt function that will make input_ids, 
            # add padding/truncate, add special tokens, + make attention masks 
            encoded_dict = tok.encode_plus(
                            sent[0],                      
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
            segment_id = [0] * 128
            segment_id[sent[2]] = 1
            segment_ids.append(segment_id)

            # This attention mask simply differentiates padding from non-padding.
            attention_masks.append(encoded_dict['attention_mask'])

    return input_ids, attention_masks, segment_ids


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

# def save_torch_model(model, optimizer):
#     # save
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, output_model)
    
def decode_result(encoded_sequence):
    
    # load tokenizer
    if (args.transformer_model).split("-")[0] == 'bert':
        if 'uncased' in args.transformer_model.split("-"):
            tok = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        else:
            tok = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
    elif (args.transformer_model).split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
    elif (args.transformer_model).split("-")[0] == 'albert':
        tok = AlbertTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        tok = XLNetTokenizer.from_pretrained(args.transformer_model, do_lower_case=False)
    
    # decode + remove special tokens
    decoded_sequence = [w for w in list(tok.convert_ids_to_tokens(encoded_sequence))
                        if not '[' in w if not '<' in w]
    
    # ' '.join(tok.convert_ids_to_tokens(encoded_sequence))\
    #     .replace('[PAD]', '').replace('[CLS]', '').replace('<pad>', '').strip()
    
    return ' '.join(decoded_sequence)