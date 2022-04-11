import torch
import os
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, \
AlbertForSequenceClassification, XLNetForSequenceClassification, CamembertForSequenceClassification, \
FlaubertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.autograd import Variable
import random
from utils import read_sents, tokenize_and_pad, format_time, flat_accuracy, decode_result
from configure import parse_args
import numpy as np
import time

if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

args = parse_args()

# argparse doesn't deal in absolutes, so we pass a str flag & convert
if args.verb_segment_ids == 'yes':
    use_segment_ids = True
else:
    use_segment_ids = False
    
print('Uses verb segment ids: ' + args.verb_segment_ids)
print('Model: ' + args.transformer_model)

if torch.cuda.is_available():      
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transformer_model = args.transformer_model
epochs = args.num_epochs

# read sentences, choose labels of telicity/duration
train_sentences, train_labels, val_sentences, val_labels, \
    test_sentences, test_labels = read_sents(args.data_path, args.label_marker)

# make input ids, attention masks, segment ids
train_inputs, train_masks, train_segments = tokenize_and_pad(train_sentences)
val_inputs, val_masks, val_segments = tokenize_and_pad(val_sentences)
test_inputs, test_masks, test_segments = tokenize_and_pad(test_sentences)

print('Train set: ' + str(len(train_inputs)))
print('Validation set: ' + str(len(val_inputs)))
print('Test set: ' + str(len(test_inputs)))

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

# Create the DataLoader for our training set.
batch_size = args.batch_size

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


if args.training == 'yes':
    if (args.transformer_model).split("-")[0] == 'bert':
        model = BertForSequenceClassification.from_pretrained(
            transformer_model,
            num_labels = 2, 
            output_attentions = True, 
            output_hidden_states = False, 
        )
    elif (args.transformer_model).split("-")[0] == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(
            transformer_model, 
            num_labels = 2,   
            output_attentions = True,
            output_hidden_states = False, 
        )
    elif (args.transformer_model).split("-")[0] == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(
            transformer_model, 
            num_labels = 2,   
            output_attentions = True,
            output_hidden_states = False,
        )
    elif (args.transformer_model).split("-")[0] == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained(
            transformer_model, 
            num_labels = 2,   
            output_attentions = True, 
            output_hidden_states = False, 
        )
    elif 'flaubert' in args.transformer_model:
        model = FlaubertForSequenceClassification.from_pretrained(
            transformer_model, 
            num_labels = 2,   
            output_attentions = True, 
            output_hidden_states = False, 
        )
    elif 'camembert' in args.transformer_model:
        model = CamembertForSequenceClassification.from_pretrained(
            transformer_model, 
            num_labels = 2,   
            output_attentions = True,
            output_hidden_states = False, 
        )

    if torch.cuda.is_available():  
        model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr = 2e-5,
                      eps = 1e-8
                    )
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []

    for epoch_i in range(0, epochs):
        print('\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                print('\tBatch {:>5,}\tof\t{:>5,}.\t\tElapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            if use_segment_ids:
                b_segments = batch[3].to(device)
            model.zero_grad()        

            if use_segment_ids:
                outputs = model(b_input_ids, 
                                token_type_ids=b_segments, 
                                attention_mask=b_input_mask, 
                                labels=b_labels, 
                                output_hidden_states=True)
            else:
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels,
                                output_hidden_states=True)
            
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)            
        
        loss_values.append(avg_train_loss)
        
        print('\tAverage training loss: {0:.2f}'.format(avg_train_loss))
        print('\tTraining epoch took: {:}'.format(format_time(time.time() - t0)))
            
        # ========================================
        #               Validation
        # ========================================
        print('\tRunning Validation...')

        t0 = time.time()
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        all_labels = []
        all_preds = []
        
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
            
            tmp_eval_accuracy = flat_accuracy(label_ids, logits)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            all_labels += label_ids.tolist()
            all_preds += np.argmax(logits, axis=1).flatten().tolist()
            assert len(all_labels) == len(all_preds)


        print('\tAccuracy: {0:.2f}'.format(eval_accuracy/nb_eval_steps))
        print('\tValidation took: {:}'.format(format_time(time.time() - t0)))
        
        print('\tConfusion matrix:\n')
        print(classification_report(all_labels, all_preds))
            
    # Save the model on epoch 4
    output_model_dir = 'checkpoints/' + args.data_path[5:] + '/' + args.label_marker + '/' + \
                        args.transformer_model.split('/')[-1] + '_' + args.verb_segment_ids
    model.save_pretrained(output_model_dir)

# ========================================
#               TESTING
# ========================================
# Load the model of the last epoch

if (args.transformer_model).split("-")[0] == 'bert':
    model = BertForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)
    
elif (args.transformer_model).split("-")[0] == 'roberta':
    model = RobertaForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)
    
elif (args.transformer_model).split("-")[0] == 'albert':
    model = AlbertForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)
    
elif (args.transformer_model).split("-")[0] == 'xlnet':
    model = XLNetForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)
    
elif 'flaubert' in args.transformer_model:
    model = FlaubertForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)
    
elif 'camembert' in args.transformer_model:
    model = CamembertForSequenceClassification.from_pretrained(
        'checkpoints/' + args.data_path[5:] + '/' + \
        args.label_marker + '/' + args.transformer_model.split('/')[-1] + \
        '_' + args.verb_segment_ids)

if torch.cuda.is_available():  
    model.cuda()

print('Loaded model succesful. Running testing...')

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

    #assert len(all_inputs) == len(all_labels) == len(all_preds) 

# Report the accuracy, the sentences
print('Accuracy: {0:.2f}'.format(test_accuracy/nb_test_steps))
print('Confusion matrix:\n')
print(classification_report(all_labels, all_preds))

# Uncomment the following to see the decoded sentences
print('\nWrong predictions:')
counter = 0
for n, sent in enumerate(all_inputs):
    if all_labels[n] != all_preds[n]:
        counter += 1
        #sentence = decode_result(sent)
        #probs = all_probs[n]
        #print('Predicted: ', all_preds[n], '\tProbs: ', str(probs), '\tSent: ', sentence)

print(str(counter) + ' out of ' + str(len(all_inputs)))

# Uncomment to see the right predictions
print('\nRight predictions:')
#for n, sent in enumerate(all_inputs):
#    if all_labels[n] == all_preds[n]:
#        counter += 1
#        sentence = decode_result(sent)
#        print('Predicted: ', '\tProbs: ', '\tSent: ', sentence)
print(str(len(all_inputs) - counter) + ' out of ' + str(len(all_inputs)))