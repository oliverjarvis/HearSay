""" Originally created by the creators of SBERT-WK and inspired from this GitHub: 
https://github.com/BinWang28/SBERT-WK-Sentence-Embedding"""

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse
import torch
import random

from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead
import Utils.SBERTutils

class SBERT_WK_Embedding:

    def __init__(self, max_seq_length = 124, seed = 69, model_type = 'binwang/bert-base-nli', embed_method = 'dissecting', context_window_size = 2, layer_start = 4, tasks = 'dissecting'):
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.model_type = model_type
        self.embed_method = embed_method
        self.context_window_size = context_window_size
        self.layer_start = layer_start
        self.tasks = tasks
        ## Loading tokenizer and model
        
        ### Loading config files
        config = AutoConfig.from_pretrained(self.model_type, cache_dir='./cache')
        config.output_hidden_states = True
        
        ### Loading tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, cache_dir='./cache')
        
        ### Loading model
        self.model = AutoModelWithLMHead.from_pretrained(self.model_type, config=config, cache_dir='./cache')


    # -----------------------------------------------
    def get_embeddings(self, sentence):
        
        ## Setting seed
        self.set_seed()

        
        ## Preprocessing sentence
        sent_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
        features_input_ids = []
        features_mask = []
        
        ### Truncate if too long
        if len(sent_ids) > self.max_seq_length:
            sent_ids = sent_ids[:self.max_seq_length]
        sent_mask = [1] * len(sent_ids)
        
        ### Padding
        padding_length = self.max_seq_length - len(sent_ids)
        sent_ids += ([0] * padding_length)
        sent_mask += ([0] * padding_length)

        ### Length Check
        assert len(sent_ids) == self.max_seq_length
        assert len(sent_mask) == self.max_seq_length

        features_input_ids.append(sent_ids)
        features_mask.append(sent_mask)

        batch_input_ids = torch.tensor(features_input_ids, dtype=torch.long)
        batch_input_mask = torch.tensor(features_mask, dtype=torch.long)
        batch = [batch_input_ids, batch_input_mask]


        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        self.model.zero_grad()

        with torch.no_grad():
            features = self.model(**inputs)[1]

        features = [layer_emb.numpy() for layer_emb in features]
        layer_embedding = []
        for i in range(features[0].shape[0]):
            layer_embedding.append(np.array([layer_emb[i] for layer_emb in features]))

        params = vars(self)

        embed_method = SBERTutils.generate_embedding(self.embed_method, features_mask)
        embedding = embed_method.embed(params, layer_embedding)[0]
        
        return embedding

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
    
    def getEmbeddingDimension():
        return 768

