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
import sklearn
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from utils import read_friedrich_sents, tokenize_and_pad
from configure import parse_args

args = parse_args()

# read friedrich sentences
sentences, _ = read_friedrich_sents(sents_dir='annotated_friedrich')
sents = [s[0] for s in sentences]
            
# make train and validation sets
shuffle(sents)
cutoff = int(len(sents) * 0.8)
train_sents = sents[:cutoff]
val_sents = sents[cutoff:]
train_df = pd.DataFrame(train_sents, columns=['text', 'labels'])
val_df = pd.DataFrame(val_sents, columns=['text', 'labels'])

# model

num_train_epochs = args.num_epochs
model_type = args.model_type
model_name = args.transfrormer_model

model_args = ClassificationArgs(num_train_epochs, 
                                overwrite_output_dir = True,
                                output_dir = 'output_dir',
                                cache_dir = 'cache_dir',
                                reprocess_input_data = True,
                                manual_seed = 42)

model = ClassificationModel(model_type, 
                            model_name, 
                            use_cuda = torch.cuda.is_available(), 
                            num_labels = 2, 
                            args = model_args)

model.train_model(train_df, verbose=False)
result, model_outputs, wrong_preds = model.eval_model(val_df, verbose=False)


with open('output_dir/eval_results.txt', 'w+') as f:
    f.write('\nnum of epochs: ' + str(num_train_epochs))
    f.write('\nmodel: ' + model_name)
    f.write(result)
    f.write('\ncorrect predictions: ' + str((len(val_sents)-len(wrong_preds))/len(val_sents)))