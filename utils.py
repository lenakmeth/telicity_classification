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
                        except KeyError: # there is no duration annotation
                            try:
                                if sent['verbs'][verb]['telicity'] == 'telic':
                                    sentences.append((sent['sent'], sent['verbs'][verb]['verb_idx']))
                                    labels.append(1)
                            except KeyError:
                                pass
    return sentences, labels


def tokenize_and_pad(sentences, lowercase=False):
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
            segment_id[sent[1]] = 1
            segment_ids.append(segment_id)
            
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
            print(encoded_dict['attention_mask'])

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

def save_torch_model(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)
    
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