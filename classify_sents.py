#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:57:28 2020

@author: lena
"""

import sys
import io
import json
from random import shuffle
import os
import torch
import pandas as pd
#import sklearn
from simpletransformers.classification import ClassificationModel, ClassificationArgs

# read friedrich sentences

sents_dir = 'annotated_friedrich'
sents = [] 
sents_files = [os.path.join(sents_dir, x) for x in os.listdir(sents_dir) if x.endswith('.json')]

for file in sents_files:
    with io.open(file, 'r') as f:
        d = json.load(f)       
        
        # labels: telic = 0, atelic = 1, stative = 1
        for sentID, sent in d.items():
            for verb in sent['verbs']:
                try:
                    if sent['verbs'][verb]['telicity'] == 'telic':
                        sents.append([sent['sent'], 0])
                    elif sent['verbs'][verb]['telicity'] == 'atelic':
                        sents.append([sent['sent'], 1])
                except KeyError: # there is no telicity annotation
                    try:
                        if sent['verbs'][verb]['duration'] == 'stative':
                            sents.append([sent['sent'], 1])
                    except KeyError:
                        pass
            
# make train and validation sets
shuffle(sents)
cutoff = int(len(sents) * 0.8)
train_sents = sents[:cutoff]
val_sents = sents[cutoff:]
train_df = pd.DataFrame(train_sents, columns=['text', 'labels'])
val_df = pd.DataFrame(val_sents, columns=['text', 'labels'])

# model
model_args = ClassificationArgs(num_train_epochs = 5, 
                                overwrite_output_dir = True,
                                #output_dir = os.getcwd(),
                                #cache_dir = os.getcwd(),
                                reprocess_input_data = True,
                                manual_seed = 42)

model = ClassificationModel(model_type = 'bert', 
                            model_name = 'bert-base-uncased', 
                            use_cuda = torch.cuda.is_available(), 
                            num_labels = 2, 
                            args = model_args)

model.train_model(train_df, verbose=True)
result, model_outputs, wrong_preds = model.eval_model(val_df, verbose=True)
