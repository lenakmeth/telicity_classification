import torch
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
import seaborn as sns
import matplotlib.pyplot as plt

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

# PARAMETERS
transformer_model = args.transformer_model
epochs = args.num_epochs

# read friedrich sentences, choose labels of telicity/duration
train_sentences, train_labels, val_sentences, val_labels, \
    test_sentences, test_labels = read_sents(args.data_path, args.label_marker)

# make input ids, attention masks, segment ids

train_inputs, train_masks, train_segments = tokenize_and_pad(train_sentences)
val_inputs, val_masks, val_segments = tokenize_and_pad(val_sentences)
test_inputs, test_masks, test_segments = tokenize_and_pad(test_sentences)

print('\nLoaded sentences and converted.')

print('Train set: ' + str(len(train_inputs)))
print('Validation set: ' + str(len(val_inputs)))
print('Test set: ' + str(len(test_inputs)))

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

if torch.cuda.is_available():  
    model.cuda()

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

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

# This training code is based on the `run_glue.py` script here:
#https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print('\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. Don't be misled--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('\tBatch {:>5,}\tof\t{:>5,}.\t\tElapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        if use_segment_ids:
            b_segments = batch[3].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
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
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)
    
    print('\tAverage training loss: {0:.2f}'.format(avg_train_loss))
    print('\tTraining epoch took: {:}'.format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print('\tRunning Validation...')

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    all_labels = []
    all_preds = []
    
    for batch in val_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        if use_segment_ids:
            b_input_ids, b_input_mask, b_labels, b_segments = batch
        else:
            b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            if use_segment_ids:
                outputs = model(b_input_ids, 
                            token_type_ids=b_segments, 
                            attention_mask=b_input_mask)
            else:
                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(label_ids, logits)
        
        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy

        # Track the number of batches
        nb_eval_steps += 1
        
        # add the labels and the predictions for the classification
        all_labels += label_ids.tolist()
        all_preds += np.argmax(logits, axis=1).flatten().tolist()
        assert len(all_labels) == len(all_preds)

    # Report the final accuracy for this validation run.
    print('\tAccuracy: {0:.2f}'.format(eval_accuracy/nb_eval_steps))
    print('\tValidation took: {:}'.format(format_time(time.time() - t0)))
    
    print('\tConfusion matrix:\n')
    print(classification_report(all_labels, all_preds))
        
# Save the model on epoch 4

output_model = 'checkpoints/' + args.data_path + '/' + args.label_marker + '/' + \
                args.transformer_model + '_' + args.verb_segment_ids + '.pth'
torch.save(model, output_model)  

# ========================================
#               TESTING
# ========================================
# Load the model of the last epoch

output_model = 'checkpoints/' + args.data_path + '/' + args.label_marker + '/' + \
                args.transformer_model + '_' + args.verb_segment_ids + '.pth'
checkpoint = torch.load(output_model, map_location=device)

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
        sentence = decode_result(sent)
        probs = all_probs[n]
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