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

args = parse_args()
lemmatizer = WordNetLemmatizer()
# pick a layer
layer_idx = -1

logger = open('log_reg/' + args.transformer_model + '_' + args.verb_segment_ids + '_' + str(layer_idx) + '.log', 'w')

# argparse doesn't deal in absolutes, so we pass a str flag & convert
if args.verb_segment_ids == 'yes':
    use_segment_ids = True
else:
    use_segment_ids = False
    
logger.write('\nUses verb segment ids: ' + args.verb_segment_ids)
logger.write('\nModel: ' + args.transformer_model)

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

# logger.write('\nTrain set: ' + str(len(train_inputs)))
# logger.write('\nValidation set: ' + str(len(val_inputs)))
# logger.write('\nTest set: ' + str(len(test_inputs)))

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

# The DataLoader needs to know our batch size for training, so we specify it here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.

batch_size = args.batch_size

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels, train_segments)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
val_data = TensorDataset(val_inputs, val_masks, val_labels, val_segments)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
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


optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):

#     logger.write('\n\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#     logger.write('\nTraining...')
    print('\n\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('\nTraining...')

    t0 = time.time()
    total_loss = 0
    model.train()
    
    if epoch_i == epochs-1:
        features = {'all_inputs': [], 'all_labels': [], 'all_sents': [],
                    'all_verb_pos': [], 'all_lengths': [], 'all_hidden_states': []}

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('\n\tBatch {:>5,}\tof\t{:>5,}.\t\tElapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_segments = batch[3].to(device)

        model.zero_grad()        

        if use_segment_ids:
            outputs = model(b_input_ids, 
                        token_type_ids=b_segments, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        else:
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
        loss = outputs[0]
        hidden_states = outputs[-2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch_i == epochs-1: # On the last epoch, save hidden states of train
            
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
            
            hidden_states_per_sentence = {sent_idx:[] for sent_idx in range(batch_size)}
            
            hidden_layer = hidden_states[layer_idx]
            
            for sent_idx in range(batch_size):
                seq_len = features['all_lengths'][sent_idx]
                temp = hidden_layer[sent_idx, :seq_len, :]
                verb_idx = features['all_verb_pos'][sent_idx]

                hidden_states_per_sentence[sent_idx].append(temp[verb_idx])
            features['all_hidden_states'] += list(hidden_states_per_sentence.values())
            
            del hidden_states_per_sentence
            del hidden_layer
            
        del hidden_states
               
    avg_train_loss = total_loss / len(train_dataloader)            
    
    loss_values.append(avg_train_loss)
    
#     logger.write('\n\tAverage training loss: {0:.2f}'.format(avg_train_loss))
#     logger.write('\n\tTraining epoch took: {:}'.format(format_time(time.time() - t0)))
    print('\n\tAverage training loss: {0:.2f}'.format(avg_train_loss))
    print('\n\tTraining epoch took: {:}'.format(format_time(time.time() - t0)))


# make the sets for the log regression model and pad as necessary

# train
# X_train = features['all_hidden_states']
# this is the hidden state for the verb of the sent, for each sent, on layer = layer_idx
X_train = np.asarray(features['all_hidden_states'])
y_train = np.asarray(train_features['all_labels'])

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
