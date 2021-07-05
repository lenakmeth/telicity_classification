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

args = parse_args()

logger = open('untrained/' + '_'.join([args.label_marker, args.transformer_model.split('/')[-1], args.verb_segment_ids]) + '.log', 'w')

# argparse doesn't deal in absolutes, so we pass a str flag & convert
if args.verb_segment_ids == 'yes':
    use_segment_ids = True
else:
    use_segment_ids = False
    
logger.write('\nUses verb segment ids: ' + args.verb_segment_ids)
logger.write('\nModel: ' + args.transformer_model)

# if torch.cuda.is_available():      
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
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

logger.write('\nTrain set: ' + str(len(train_inputs)))
logger.write('\nValidation set: ' + str(len(val_inputs)))
logger.write('\nTest set: ' + str(len(test_inputs)))

# Convert all inputs and labels into torch tensors, the required datatype for our model.
train_inputs = torch.tensor(train_inputs)
val_inputs = torch.tensor(val_inputs)
test_inputs = torch.tensor(test_inputs)

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

if use_segment_ids:
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
if use_segment_ids:
    train_data = TensorDataset(train_inputs, train_masks, train_labels, train_segments)
else:
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
if use_segment_ids:
    val_data = TensorDataset(val_inputs, val_masks, val_labels, val_segments)
else:
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create the DataLoader for our test set.
if use_segment_ids:
    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_segments)
else:
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 

if (args.transformer_model).split("-")[0] == 'bert':
    model = BertForSequenceClassification.from_pretrained(
        transformer_model,
        num_labels = 2, # The number of output labels--2 for binary classification. 
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
elif (args.transformer_model).split("-")[0] == 'roberta':
    model = RobertaForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
elif (args.transformer_model).split("-")[0] == 'albert':
    model = AlbertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
elif (args.transformer_model).split("-")[0] == 'xlnet':
    model = XLNetForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
elif 'flaubert' in args.transformer_model:
    model = FlaubertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
elif 'camembert' in args.transformer_model:
    model = CamembertForSequenceClassification.from_pretrained(
        transformer_model, 
        num_labels = 2,   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

# if torch.cuda.is_available():  
#     model.cuda()

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('\nThe model has {:} different named parameters.\n'.format(len(params)))

print('\n==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n\n==== First Transformer ====\n')

for p in params[5:21]:
    logger.write('\n{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

print('\n\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
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

# ========================================
#               TESTING
# ========================================
# Load the model of the last epoch

logger.write('\nValidation:...')
print('\nval')

t0 = time.time()
model.eval()
val_loss, val_accuracy = 0, 0
nb_val_steps, nb_val_examples = 0, 0

all_inputs = []
all_labels = []
all_preds = []
all_probs = []
    
for batch in val_dataloader:
    batch = tuple(t.to(device) for t in batch)
    if use_segment_ids:
        b_input_ids, b_input_mask, b_labels, b_segments = batch
    else:
        b_input_ids, b_input_mask, b_labels = batch
        
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
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)
    
    
    tmp_val_accuracy = flat_accuracy(label_ids, logits)
    val_accuracy += tmp_val_accuracy
    nb_val_steps += 1
    
    all_inputs += b_input_ids.to('cpu').numpy().tolist()
    all_labels += label_ids.tolist()
    all_preds += np.argmax(logits, axis=1).flatten().tolist()
    all_probs += log_probs.tolist()
    assert len(all_labels) == len(all_preds)

# Report the accuracy, the sentences
logger.write('\nAccuracy: {0:.2f}'.format(val_accuracy/nb_val_steps))
logger.write('\nConfusion matrix:\n')
logger.write(classification_report(all_labels, all_preds))

print('\nAccuracy: {0:.2f}'.format(val_accuracy/nb_val_steps))
print('\nConfusion matrix:\n')
print(classification_report(all_labels, all_preds))

# Uncomment the following to see the decoded sentences
# change != to == to see the right predictions
logger.write('\n\nWrong predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] != all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nwrong_val: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

logger.write('\n\nRight predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] == all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nright_val: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

# ========================================
#               TESTING
# ========================================
# Load the model of the last epoch

logger.write('\nLoaded model succesful. Running testing...')
print('\ntest')

t0 = time.time()
model.eval()
test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0

all_inputs = []
all_labels = []
all_preds = []
all_probs = []
    
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    if use_segment_ids:
        b_input_ids, b_input_mask, b_labels, b_segments = batch
    else:
        b_input_ids, b_input_mask, b_labels = batch
        
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
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)
    
    
    tmp_test_accuracy = flat_accuracy(label_ids, logits)
    test_accuracy += tmp_test_accuracy
    nb_test_steps += 1
    
    all_inputs += b_input_ids.to('cpu').numpy().tolist()
    all_labels += label_ids.tolist()
    all_preds += np.argmax(logits, axis=1).flatten().tolist()
    all_probs += log_probs.tolist()
    assert len(all_labels) == len(all_preds)

# Report the accuracy, the sentences
logger.write('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
logger.write('\nConfusion matrix:\n')
logger.write(classification_report(all_labels, all_preds))

print('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
print('\nConfusion matrix:\n')
print(classification_report(all_labels, all_preds))

# Uncomment the following to see the decoded sentences
# change != to == to see the right predictions
logger.write('\n\nWrong predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] != all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nwrong_seen: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

logger.write('\n\nRight predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] == all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nright_seen: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

# ========================================
#               TESTING
# ========================================
# Load the model of the last epoch


logger.write('\nTesting unseen')
print('\ntest 2')

# open the unseen
test_sentences = []
test_labels = []

with open('data/unseen_tests/' + args.label_marker + '_unseen.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        l = line.strip().split('\t')
        test_sentences.append([l[-4], l[-3], int(l[-2])])
        test_labels.append(int(l[-1]))

test_inputs, test_masks, test_segments = tokenize_and_pad(test_sentences)

test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)
if use_segment_ids:
    test_segments = torch.tensor(test_segments)

# Create the DataLoader for our test set.
if use_segment_ids:
    test_data = TensorDataset(test_inputs, test_masks, test_labels, test_segments)
else:
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

t0 = time.time()
model.eval()
test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0

all_inputs = []
all_labels = []
all_preds = []
all_probs = []
    
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    if use_segment_ids:
        b_input_ids, b_input_mask, b_labels, b_segments = batch
    else:
        b_input_ids, b_input_mask, b_labels = batch
        
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
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)
    
    
    tmp_test_accuracy = flat_accuracy(label_ids, logits)
    test_accuracy += tmp_test_accuracy
    nb_test_steps += 1
    
    all_inputs += b_input_ids.to('cpu').numpy().tolist()
    all_labels += label_ids.tolist()
    all_preds += np.argmax(logits, axis=1).flatten().tolist()
    all_probs += log_probs.tolist()
    assert len(all_labels) == len(all_preds)

# Report the accuracy, the sentences
logger.write('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
logger.write('\nConfusion matrix:\n')
logger.write(classification_report(all_labels, all_preds))

print('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
print('\nConfusion matrix:\n')
print(classification_report(all_labels, all_preds))

# Uncomment the following to see the decoded sentences
# change != to == to see the right predictions
logger.write('\n\nWrong predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] != all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nwrong_unseen: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

logger.write('\n\nRight predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] == all_preds[n]:
        counter += 1
        sentence = decode_result(sent)
        logger.write('\nright_unseen: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
        logger.write(sentence)
logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

# ========================================
#               TESTING
# ========================================


if args.label_marker == 'telicity':

    logger.write('\nTesting min pairs')
    print('\ntest 3')

    # open the unseen
    test_sentences = []
    test_labels = []

    with open('data/unseen_tests/' + args.label_marker + '_min_pairs.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip().split('\t')
            test_sentences.append([l[-4], l[-3], int(l[-2])])
            test_labels.append(int(l[-1]))

    test_inputs, test_masks, test_segments = tokenize_and_pad(test_sentences)

    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)
    if use_segment_ids:
        test_segments = torch.tensor(test_segments)

    # Create the DataLoader for our test set.
    if use_segment_ids:
        test_data = TensorDataset(test_inputs, test_masks, test_labels, test_segments)
    else:
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    t0 = time.time()
    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    all_inputs = []
    all_labels = []
    all_preds = []
    all_probs = []
        
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        if use_segment_ids:
            b_input_ids, b_input_mask, b_labels, b_segments = batch
        else:
            b_input_ids, b_input_mask, b_labels = batch
            
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
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        log_probs = F.softmax(Variable(torch.from_numpy(logits)), dim=-1)
        
        
        tmp_test_accuracy = flat_accuracy(label_ids, logits)
        test_accuracy += tmp_test_accuracy
        nb_test_steps += 1
        
        all_inputs += b_input_ids.to('cpu').numpy().tolist()
        all_labels += label_ids.tolist()
        all_preds += np.argmax(logits, axis=1).flatten().tolist()
        all_probs += log_probs.tolist()
        assert len(all_labels) == len(all_preds)

    # Report the accuracy, the sentences
    logger.write('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
    logger.write('\nConfusion matrix:\n')
    logger.write(classification_report(all_labels, all_preds))

    print('\nAccuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
    print('\nConfusion matrix:\n')
    print(classification_report(all_labels, all_preds))

    # Uncomment the following to see the decoded sentences
    # change != to == to see the right predictions
    logger.write('\n\nWrong predictions:')
    counter = 0
    for n, sent in enumerate(all_inputs):
        if all_labels[n] != all_preds[n]:
            counter += 1
            sentence = decode_result(sent)
            logger.write('\nwrong_min: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
            logger.write(sentence)
    logger.write(str(counter) + ' out of ' + str(len(all_inputs)))

    logger.write('\n\nRight predictions:')
    counter = 0
    for n, sent in enumerate(all_inputs):
        if all_labels[n] == all_preds[n]:
            counter += 1
            sentence = decode_result(sent)
            logger.write('\nright_min: ' + str(all_preds[n]) + '\tprob: ' + str(all_probs[n])+ '\t')
            logger.write(sentence)
    logger.write(str(counter) + ' out of ' + str(len(all_inputs)))