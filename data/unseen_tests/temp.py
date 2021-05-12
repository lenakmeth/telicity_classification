# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import spacy

nlp = spacy.load("fr_core_news_sm")
files = [f for f in os.listdir(os.getcwd()) if f.startswith('FR')]



for file in files:
    sents = []
    labels = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            sents.append(line.split('\t')[0])
            labels.append(line.split('\t')[1].strip())
            
            
            
    new_sents = []
    
    for sent in sents:
        
        temp = []
        verb = False
        verb_idx = False
        
        doc = nlp(sent)

        for n, token in enumerate(doc):
            temp.append(token.text)
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                verb = token.text
                verb_idx = n
        if not verb:
            for n, token in enumerate(doc):
                if token.dep_ == 'ROOT':
                    verb = token.text
                    verb_idx = n 
                    
        new_sents.append([' '.join(temp), verb, str(verb_idx)])
    
    with open('french' + file[2:], 'w') as f:
        for n, sent in enumerate(new_sents):
            f.write('\t'.join(sent))
            f.write('\t' + labels[n] + '\n')
            