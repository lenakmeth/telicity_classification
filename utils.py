import io
import json
import os
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, XLMRobertaTokenizer
from configure import parse_args
import time
import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix  

args = parse_args()

def read_friedrich_sents(sents_dir, marker):
    """ Read the .json files with the annotated sentences. 
        NB: The sentences elements are a TUPLE of (sentence, verb index)."""
   
    sentences = [] 
    labels = []
    sents_files = [os.path.join(sents_dir, x) for x in os.listdir(sents_dir) if x.endswith('.json')]

    for file in sents_files:
        with io.open(file, 'r') as f:
            d = json.load(f)       
            
            # labels: telic = 0, atelic = 1, stative = 1
            for sentID, sent in d.items():                
                for verb in sent['verbs']:
                    
                    if marker == 'telicity':
                        try:
                            if sent['verbs'][verb]['telicity'] == 'telic':
                                sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                labels.append(0)
                            elif sent['verbs'][verb]['telicity'] == 'atelic':
                                sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                labels.append(1)
                        except KeyError: # there is no telicity annotation
                            try:
                                if sent['verbs'][verb]['duration'] == 'stative':
                                    sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                    labels.append(1)
                            except KeyError:
                                pass
                            
                    elif marker == 'duration':
                        try:
                            if sent['verbs'][verb]['duration'] == 'stative':
                                sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                labels.append(0)
                            elif sent['verbs'][verb]['duration'] == 'dynamic':
                                sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                labels.append(1)
                        except KeyError: # there is no telicity annotation
                            try:
                                if sent['verbs'][verb]['telicity'] == 'telic':
                                    sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                    labels.append(1)
                            except KeyError:
                                pass
    return sentences, labels


def tokenize_and_pad(sentences):
    """ We are using .encode_plus. This does not make specialized attn masks 
        like in our selectional preferences experiment. Revert to .encode if
        necessary."""
    
    input_ids = []
    segment_ids = [] # token type ids
    attention_masks = []
    
    
    if (args.transformer_model).split("-")[0] == 'bert':
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = BertTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        
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
            segment_id[sent[1]] = 1
            segment_ids.append(segment_id)

            # This attention mask simply differentiates padding from non-padding.
            attention_masks.append(encoded_dict['attention_mask'])
    
    # ======== RoBERTa ========
    elif (args.transformer_model).split("-")[0] == 'roberta':
        tok = RobertaTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        
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
        tok = AlbertTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        
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
            segment_id[sent[1]] = 1
            segment_ids.append(segment_id)

            # This attention mask simply differentiates padding from non-padding.
            attention_masks.append(encoded_dict['attention_mask'])

    # ======== XLNET ======== 
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        # Tokenize all of the sentences and map the tokens to their word IDs.
        tok = XLNetTokenizer.from_pretrained(args.transformer_model, do_lower_case=True)
        
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
            segment_id[sent[1]] = 1
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
