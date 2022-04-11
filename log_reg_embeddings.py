import torch
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, \
AlbertForSequenceClassification, XLNetForSequenceClassification, CamembertForSequenceClassification, \
FlaubertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
from utils import *
import torch.nn.functional as F
from torch.autograd import Variable
from configure import parse_args
import numpy as np
import time
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

args = parse_args()
lemmatizer = WordNetLemmatizer()
# pick a layer
layer_idx = args.freeze_layer_count

# argparse doesn't deal in absolutes, so we pass a str flag & convert
if args.verb_segment_ids == 'yes':
    use_segment_ids = True
else:
    use_segment_ids = False

print('\nModel: ' + args.transformer_model)
print('\nUses verb segment ids: ' + args.verb_segment_ids)
print('\nModel: ' + args.transformer_model)
print('\nUses verb segment ids: ' + args.verb_segment_ids)

device = torch.device("cpu")

# PARAMETERS
transformer_model = args.transformer_model
epochs = args.num_epochs

# read friedrich sentences, choose labels of telicity/duration
train_sentences, train_labels, val_sentences, val_labels, \
    test_sentences, test_labels = read_sents(args.data_path, args.label_marker)

# make input ids, attention masks, segment ids, depending on the model we will use

train_inputs, train_masks, train_segments = tokenize_and_pad(train_sentences)
val_inputs, val_masks, val_segments = tokenize_and_pad(val_sentences)
test_inputs, test_masks, test_segments = tokenize_and_pad(test_sentences)

print('\n\nLoaded sentences and converted.')

# Convert all inputs and labels into torch tensors, the required datatype for our model.
train_inputs = torch.tensor(train_inputs)
val_inputs = torch.tensor(val_inputs)
test_inputs = torch.tensor(test_inputs)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

train_segments = torch.tensor(train_segments)
val_segments = torch.tensor(val_segments)
test_segments = torch.tensor(test_segments)

train_masks = torch.tensor(train_masks)
val_masks = torch.tensor(val_masks)
test_masks = torch.tensor(test_masks)

# DataLoader
batch_size = args.batch_size

train_data = TensorDataset(train_inputs, train_masks, train_labels, train_segments)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels, val_segments)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels, test_segments)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

if (args.transformer_model).split("-")[0] == 'bert':
    model = BertForSequenceClassification.from_pretrained(
        transformer_model,
        num_labels = 2, 
        output_attentions = True,
        output_hidden_states = True, 
    )
elif (args.transformer_model).split("-")[0] == 'roberta':
    model = RobertaForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = True,
        output_hidden_states = True,
    )
elif (args.transformer_model).split("-")[0] == 'albert':
    model = AlbertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = True,
        output_hidden_states = True, 
    )
elif (args.transformer_model).split("-")[0] == 'xlnet':
    model = XLNetForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = True,
        output_hidden_states = True,
    )
elif 'flaubert' in args.transformer_model:
    model = FlaubertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = True,
        output_hidden_states = True,
    )
elif 'camembert' in args.transformer_model:
    model = CamembertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = True, 
        output_hidden_states = True, 
    )


# GET EMBEDDINGS

def get_features(dataloader):

    model.eval()
    val_loss, val_accuracy = 0, 0
    nb_val_steps, nb_val_examples = 0, 0

    features = {'all_inputs': [], 'all_labels': [], 'all_sents': [],
                        'all_verb_pos': [], 'all_lengths': [], 'all_hidden_states': []}

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_segments = batch


        with torch.no_grad():        
            if use_segment_ids:
                outputs = model(b_input_ids, 
                                token_type_ids=b_segments, 
                                attention_mask=b_input_mask)
            else:
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)

        logits = outputs[0]
        hidden_states = outputs[-2]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)

        all_inputs = b_input_ids.to('cpu').numpy().tolist()
        features['all_inputs'] += all_inputs

        all_segments = b_segments.to('cpu').numpy().tolist()
        features['all_verb_pos'] += [vector.index(1) for vector in all_segments]

        label_ids = b_labels.to('cpu').numpy().tolist()
        features['all_labels'] += label_ids

        for n, input_sent in enumerate(all_inputs):
            decoded_sentence = decode_result(input_sent)
            features['all_sents'].append(decoded_sentence)
            features['all_lengths'].append(len(decoded_sentence.split(' ')))

        hidden_layer = hidden_states[layer_idx]

        for sent_idx in range(len(all_inputs)):
            try:
                seq_len = features['all_lengths'][sent_idx]
                temp = hidden_layer[sent_idx, :seq_len, :]
                verb_idx = features['all_verb_pos'][sent_idx]

                features['all_hidden_states'].append(np.asarray(temp[verb_idx]))
            except IndexError:
                print(hidden_layer.shape)
                print(features['all_lengths'][sent_idx])

    return features

# make the sets for the log regression model and pad as necessary
train_features = get_features(train_dataloader)
val_features = get_features(val_dataloader)
test_features = get_features(test_dataloader)

X_train = np.asarray(train_features['all_hidden_states'])
y_train = np.asarray(train_features['all_labels'])

X_val = np.asarray(val_features['all_hidden_states'])
y_val = np.asarray(val_features['all_labels'])

X_test = np.asarray(train_features['all_hidden_states'])
y_test = np.asarray(train_features['all_labels'])

# train log reg
logit = LogisticRegression(C=1e-2, random_state=17, solver='lbfgs', 
                           multi_class='multinomial', max_iter=100,
                          n_jobs=4)

logit.fit(X_train, y_train)

print('layer:', layer_idx)

logit_val_pred = logit.predict_proba(X_val)
result_val = log_loss(y_val, logit_val_pred)
print(result_val)

logit_test_pred = logit.predict_proba(X_test)
result_test = log_loss(y_test, logit_test_pred)
print(result_test)

print('\nlayer: ' + str(layer_idx))
print('\nval: ' + str(result_val))
print('\ntest: ' + str(result_test))