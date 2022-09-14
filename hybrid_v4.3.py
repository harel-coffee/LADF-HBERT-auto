#!/usr/bin/env python
# coding: utf-8

# ## Part 1 Build on word-level text to generate a fixed-length vector for each sentence


import copy, itertools, math

import random as r
r.seed(2019)

import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)

from util import d, here

import pandas as pd
from argparse import ArgumentParser
from datetime import datetime

import random, sys, math, gzip
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

import numpy as np
#np.random.seed(2019)
np.set_printoptions(threshold=sys.maxsize)
from numpy import genfromtxt

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.utils import shuffle
import glob
from optparse import OptionParser

import pickle
import shutil

## Importing ADF python files (should be kept in the base directory).
import ascribe_ADF

args = {

    ## directory for encoded embeddings
    'data_dir': 'datasets/roberta-base_data/',

    ## output of the experiments
    'output_dir': 'outputs/',

    ## output of attention scores for sentences
    'attention_output_dir': 'attentions/',

    'cuda_num': 1,
  
}


 ## take a list of strings as optional arguments
def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)


class Normalize(object):
    
    def normalize_train(self, X_train, max_len):

        self.scaler = MinMaxScaler()
        X_train = X_train.reshape(X_train.shape[0],-1)
        X_train = self.scaler.fit_transform(X_train)
        
        X_train = X_train.reshape(X_train.shape[0],max_len,-1)
       
        return X_train
    
    def normalize_test(self, X_test, max_len):
    
        X_test = X_test.reshape(X_test.shape[0],-1)
        X_test = self.scaler.transform(X_test)
        
        X_test = X_test.reshape(X_test.shape[0],max_len,-1)
        
        return X_test

    def inverse(self, X_train, X_test):

        X_train = self.scaler.inverse_transform(X_train)
        X_test   = self.scaler.inverse_transform(X_test)
    
        return (X_train, X_test)


# ## Part 1. Import sentences embeddings


def import_data(dataset, max_len):

    with open(os.path.join(args['data_dir'], dataset + '_non.p'), 'rb') as fp:
        pre_trained_non_dict = pickle.load(fp)
        
    with open(os.path.join(args['data_dir'], dataset + '_vio.p'), 'rb') as fp:
        pre_trained_vio_dict = pickle.load(fp)

    pre_trained_non = [pre_trained_non_dict[i] for i in pre_trained_non_dict.keys()]
    pre_trained_vio = [pre_trained_vio_dict[i] for i in pre_trained_vio_dict.keys()]

    # Padding

    vio = np.zeros((len(pre_trained_vio), max_len, 768)) 
    non = np.zeros((len(pre_trained_non), max_len, 768))    

    # assigning values
    for idx, doc in enumerate(pre_trained_vio):

        if doc.shape[0] <= max_len:
            vio[idx][:doc.shape[0], :] = doc

        else:
            vio[idx][:max_len, :] = doc[:max_len, :]
            
    print('violation example shape: ', vio.shape)
    
    # assert np.array_equal(vio[0],pre_trained_vio[0][:max_len,:])

    for idx,doc in enumerate(pre_trained_non): 

        if doc.shape[0] <= max_len:
            non[idx][:doc.shape[0], :] = doc

        else:
            non[idx][:max_len,:] = doc[:max_len, :]
            
    print('nonviolation example shape: ', non.shape)
    print('nonviolation example size', sizeof_fmt(sys.getsizeof(non)))
    
    return non, vio


# ## Part 2. Config model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class BertConfig():

    """
        :class:`~transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.
        Arguments:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            seq_length: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=4,
                 num_attention_heads=1,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.01,
                 attention_probs_dropout_prob=0.01,
                 seq_length=512,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 output_attentions=True,
                 output_hidden_states=False,
                 num_labels=2):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.seq_length = seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels


# ## Part 3. Customized Sentence-level Transformer


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):

    """input sentence embeddings inferred by bottom pre-trained BERT, contruct location embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        
        self.config = config
        self.location_embeddings = nn.Embedding(config.seq_length, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, inputs_embeds, location_ids=None):
    
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device
        
        if location_ids is None:
            location_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            location_ids = location_ids.unsqueeze(0).expand(input_shape)

        location_embeddings = self.location_embeddings(location_ids)
        
        embeddings = inputs_embeds + location_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    
    def __init__(self, config):

        super(BertSelfAttention, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)


    def forward(self, hidden_states, attention_mask=None):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)

        return outputs


class BertSelfOutput(nn.Module):

    def __init__(self, config):

        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertAttention(nn.Module):

    def __init__(self,config):

        super(BertAttention, self).__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None):
 
        self_outputs = self.self(hidden_states, attention_mask)
  
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):

        super(BertIntermediate, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        self.intermediate_act_fn = torch.nn.ReLU()


    def forward(self, hidden_states):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):

        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, hidden_states, input_tensor):

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):

        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


    def forward(self, hidden_states, attention_mask=None):

        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class BertEncoder(nn.Module):

    def __init__(self, config):

        super(BertEncoder, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])


    def forward(self, hidden_states, attention_mask=None):

        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        if self.output_attentions:
            outputs = outputs + (all_attentions,)
                    
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):

    def __init__(self, config):

        super(BertPooler, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


    def forward(self, hidden_states):

        # We "pool" the model by simply averaging hidden states
        mean_tensor = hidden_states.mean(dim=1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(nn.Module):

    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        
    """  
    
    def __init__(self, config):

        super(BertModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
 

    def forward(self, inputs_embeds, attention_mask=None, location_ids=None):

        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
            
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
       
        input_shape = inputs_embeds.size()[:-1]

        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        
            extended_attention_mask = attention_mask[:, None, None, :]

        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for locations we want to attend and 0.0 for
        # masked locations, this operation will create a tensor which is 0.0 for
        # locations we want to attend and -10000.0 for masked locations.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
   
        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, location_ids=location_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class HTransformer(nn.Module):

    """
    Sentence-level transformer, several transformer blocks + softmax layer
    
    """

    def __init__(self, config):

        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.
     
        """
        super(HTransformer,self).__init__()
        
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)


    def forward(self, x):

        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            location_ids=None,
                            inputs_embeds=x)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
    
        outputs = (logits,) + outputs[2:]        

        return outputs


def init_weights(module):

    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)

    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def train_embeddings(non_list, vio_list, max_len):

    ## Balancing data.
    non_array = np.asarray(non_list)
    vio_array = np.asarray(vio_list)

    non_array_len = len(non_array)
    vio_array_len = len(vio_array)
    
    ## Setting up training embeddings.
    if non_array_len > 0:

        if vio_array_len > 0:

            train_text_emb = np.concatenate((non_array, vio_array), axis = 0)
            assert train_text_emb.shape == (non_array_len + vio_array_len, max_len, 768)

            train_label_emb = np.array([0] * non_array_len + [1] * vio_array_len)

        else:

            train_text_emb = non_array
            assert train_text_emb.shape == (non_array_len, max_len, 768)

            train_label_emb = np.array([0] * non_array_len)

    else:

        if vio_array_len > 0:

            train_text_emb = vio_array
            assert train_text_emb.shape == (vio_array_len, max_len, 768)

            train_label_emb = np.array([1] * vio_array_len)

        else:

            train_text_emb = []
            train_label_emb = []

    del non_array, vio_array
    
    return train_text_emb, train_label_emb


def factorascribe(text, model, max_len, dataset, fidx, baselabs, batch, base_factors):

    text_array = np.asarray(text)
    norm_factor = normalizer.normalize_test(text_array, max_len)

    tensor_factor_x = torch.from_numpy(norm_factor).type(torch.FloatTensor)
    factor_set = torch.utils.data.TensorDataset(tensor_factor_x)
    factorloader = torch.utils.data.DataLoader(factor_set, batch_size = batch, shuffle = False, num_workers = 1)

    with torch.no_grad():
        
        y_predict = []

        model.train(False)

        atten_scores = torch.Tensor()
        
        for idx, data in enumerate(tqdm(factorloader)):

            inputs = data[0]

            if inputs.size(1) > config.seq_length:
                inputs = inputs[:, :config.seq_length, :]

            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda(1))
            
            out = model(inputs)
            
            sm = nn.Softmax(dim = 1)
            pred_prob = out[0].cpu()
            pred_prob = sm(pred_prob) 
            """
            if config.output_attentions:
            
                last_layer_attention = out[1][-1].cpu()
                atten_scores = torch.cat((atten_scores, last_layer_attention))
            """
            predict = torch.argmax(pred_prob, axis = 1)
            
            y_predict += predict

        assert len(y_predict) == len(text)

        for elemidx, elem in enumerate(y_predict):
                
            change = False
            assert elem in [1, 0]

            if elem == 1:

                baselabs[elemidx][-(base_factors - fidx)] = "in"
                change = True

            elif elem == 0:

                baselabs[elemidx][-(base_factors - fidx)] = "off"
                change = True

            assert change

    return baselabs


def conv_label(verdict):

    assert verdict in ["in", "out", "off"]
      
    if verdict == "in":
        outlab = 1
    
    elif verdict in ["out", "off"]:
        outlab = 0
    
    return outlab


def gen_doc_weights(didx, fidx, train_non, train_vio, docs_class_non, docs_class_vio):

    train_non_len = len(train_non)
    train_vio_len = len(train_vio)

    if train_non_len >= train_vio_len:

        if didx < train_non_len:

            non_prob = docs_class_non[didx][fidx]
            vio_prob = docs_class_vio[didx][fidx]

            document = train_non[didx]

        else:
        
            mod_idx = divmod(didx - train_non_len, train_vio_len)[1]

            non_prob = docs_class_non[mod_idx + train_non_len][fidx]
            vio_prob = docs_class_vio[mod_idx + train_non_len][fidx]

            document = train_vio[mod_idx]

    else:

        if didx < train_vio_len:

            mod_idx = divmod(didx, train_non_len)[1]

            non_prob = docs_class_non[mod_idx][fidx]
            vio_prob = docs_class_vio[mod_idx][fidx]

            document = train_non[mod_idx]

        else:

            non_prob = docs_class_non[didx - train_vio_len + train_non_len][fidx]
            vio_prob = docs_class_vio[didx - train_vio_len + train_non_len][fidx]

            document = train_vio[didx - train_vio_len]

    return non_prob, vio_prob, document


def eval_stats(eval_non_len, eval_vio_len, eval_non_baselabs, eval_vio_baselabs):
    
    y_eval_pred = []
    y_eval_true = [0] * eval_non_len + [1] * eval_vio_len

    for idx in range(eval_non_len):

        labs, non_layers = ADF.enumlabtype("stable", False, eval_non_baselabs[idx])
        y_eval_pred.append(conv_label(labs[0]))

    for idx in range(eval_vio_len):

        labs, vio_layers = ADF.enumlabtype("stable", True, eval_vio_baselabs[idx])
        y_eval_pred.append(conv_label(labs[0]))

    return y_eval_true, y_eval_pred


if __name__ == "__main__":
    
    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -l learning_rate -e no_epochs -m max_len')
    
    parser.add_option("-d", "--dataset_name", action = "store", type = "string", dest = "dataset_name", help = "directory of data encoded by token-level Roberta", default = 'article6')
    parser.add_option("-l", "--learning_rate", action = "store", type = "float", dest = "learning_rate", help = "learning rate for fine tuning", default = 2e-6)
    parser.add_option("-e", "--no_epochs", action = "store", type = "int", dest = "no_epochs", help = "the number of epochs for fine tuning", default = 30)
    parser.add_option("-m", "--max_len", action = "store", type = "int", dest = "max_len", help = "the maximum number of bullets per document", default = 256)
    parser.add_option('-r', "--random_seed", type = 'int', action = 'callback', dest = 'random_seed', callback = list_callback, default = 1988)
    parser.add_option('-p', "--test_prop", type = 'float', action = 'store', dest = 'test_prop', help = "the proportion of the data used for training", default = 0.2)

    (options, _) = parser.parse_args()
    
    dataset = options.dataset_name
    lr = options.learning_rate
    no_epochs = options.no_epochs
    max_len = options.max_len
    seed = options.random_seed
    test_prop = options.test_prop

    script_name = os.path.basename(__file__)[:-3]
    save_name = '%s_epoch%s_prop%s_len%s/'%(script_name, no_epochs, test_prop, max_len)

    unique_id = datetime.today().strftime('%Y-%m-%d-%Hhr%M')
    print("unique_id: ", unique_id)
    
    print('dataset name: ', dataset)
    print('number of epochs: ', no_epochs)
    print('initial random state: ', seed)
    print("proportion of data for test: ", test_prop)
    print("max no. of bullets: ", max_len)

    gradient_clipping = 1.0
    train_batch = 16 
    eval_batch = 32
    
    ## Use ascribe_ADF.py to read the relevant ADF csv files to create the
    ## labelling and error propagation scaffolding used to ascribe factors and
    ## pass back the classification weights to the NLP task.
    ## Setting number of base-level factors for classification/prediction in NLP task.
    assert dataset == "article6"
    admissible_factors = False

    if admissible_factors:

        ADF = ascribe_ADF.AF(dataset + ".csv")
        base_factors = 35

    else:
    
        ADF = ascribe_ADF.AF("art6_counts.csv")
        base_factors = 32

    data_non, data_vio = import_data(dataset, max_len)

    ## Setting data set to January 2015 data
    data_non = data_non[:150]
    data_vio = data_vio[:425]
    """
    ## Setting data set to January 2001 data
    data_non = data_non[:396]
    data_vio = data_vio[:2451]
    """
    config = BertConfig(seq_length = max_len)

    ## Setting up training and test data sets.
    learn_non, test_non = train_test_split(data_non, test_size = test_prop)
    learn_vio, test_vio = train_test_split(data_vio, test_size = test_prop)

    learn_non_len = len(learn_non)
    print("learn_non_len: ", learn_non_len)
    learn_vio_len = len(learn_vio)
    print("learn_vio_len: ", learn_vio_len)

    set_size = learn_non_len + learn_vio_len

    test_non_len = len(test_non)
    print("test_non_len: ", test_non_len)
    test_vio_len = len(test_vio)
    print("test_vio_len: ", test_vio_len)

    ## Setting number of base-level factors for classification/prediction in NLP task.
    val_prop = 0.1
    assert dataset == "article6"   
    learn_non_baselabs = [["undec"] * (ADF.depth - base_factors) + ["fix_undec"] * base_factors for _ in range(learn_non_len)]
    learn_vio_baselabs = [["undec"] * (ADF.depth - base_factors) + ["fix_undec"] * base_factors for _ in range(learn_vio_len)]
    
    for fidx in range(base_factors):

        base_model = HTransformer(config = config)
        base_model.apply(init_weights)

        with open("model_%s.p"%(fidx), 'wb') as fp:
            pickle.dump(base_model, fp, pickle.HIGHEST_PROTOCOL)

    ## Initialising analytics
    train_factor_ascribe = [[] for _ in range(base_factors)]
    train_confusion_matrices = []
    train_mccs = []
    
    test_factor_ascribe = [[] for _ in range(base_factors)]
    test_confusion_matrices = []
    test_mccs = []

    losses = []
    macro_f = []

    train_best_mcc = 0.0
    test_best_mcc = 0.0

    print("Setup completed. Ready to start!")

    for e in tqdm(range(no_epochs)):

        ## Settting validation set.
        train_non, val_non, non_baselabs, val_non_baselabs = train_test_split(learn_non, learn_non_baselabs, test_size = val_prop)
        train_vio, val_vio, vio_baselabs, val_vio_baselabs = train_test_split(learn_vio, learn_vio_baselabs, test_size = val_prop)

        train_non_len = len(train_non)
        train_vio_len = len(train_vio)
        val_non_len = len(val_non)
        val_vio_len = len(val_vio)
        
        print('\n epoch ',e)
        
        ## Setting for only the base-level factors.
        docs_class_non = np.zeros((train_non_len + train_vio_len, base_factors))
        docs_class_vio = np.zeros((train_non_len + train_vio_len, base_factors))
        
        for docidx, doc in enumerate(train_non):
            
            non_lab, non_layers = ADF.enumlabtype("stable", False, non_baselabs[docidx])

            if non_lab[0] == "out":

                temp_factors_non = ADF.weightvector[0][-base_factors:]
                temp_factors_vio = ADF.weightvector[1][-base_factors:]

                assert len(temp_factors_non) == len(temp_factors_vio) == base_factors

                origin_base = ADF.depth - base_factors

                for bidx in range(base_factors):

                    tree_scale_non = 1 - temp_factors_non[bidx]
                    tree_scale_vio = 1 - temp_factors_vio[bidx]

                    assert non_lab[origin_base + bidx] in ["in", "off"]
                    if non_lab[origin_base + bidx] ==  "off":
                        
                        if val_epoch_mcc > 0:
                            lab_scale_non = 1 - val_epoch_mcc

                        else:
                            lab_scale_non = 1.0

                        lab_scale_vio = 1.0

                    else:

                        if val_epoch_mcc > 0:
                            lab_scale_vio = 1 - val_epoch_mcc

                        else:
                            lab_scale_vio = 1.0

                        lab_scale_non = 1.0

                    docs_class_non[docidx][bidx] = 1 - tree_scale_non * lab_scale_non
                    docs_class_vio[docidx][bidx] = 1 - tree_scale_vio * lab_scale_vio

            else:

                temp_factors_non = ADF.weightvector[0][-base_factors:]
                temp_factors_vio = ADF.weightvector[1][-base_factors:]

                for bidx in range(base_factors):

                    docs_class_non[docidx][bidx] = temp_factors_non[bidx]
                    docs_class_vio[docidx][bidx] = temp_factors_vio[bidx]
        
        print("Neg documents processed")

        for docidx, doc in enumerate(train_vio):
            
            vio_lab, vio_layers = ADF.enumlabtype("stable", True, vio_baselabs[docidx])

            if vio_lab[0] == "in":

                temp_factors_non = ADF.weightvector[0][-base_factors:]
                temp_factors_vio = ADF.weightvector[1][-base_factors:]

                assert len(temp_factors_non) == len(temp_factors_vio) == base_factors

                origin_base = ADF.depth - base_factors

                for bidx in range(base_factors):

                    tree_scale_non = 1 - temp_factors_non[bidx]
                    tree_scale_vio = 1 - temp_factors_vio[bidx]

                    assert vio_lab[origin_base + bidx] in ["in", "off"]
                    if vio_lab[origin_base + bidx] ==  "off":

                        if val_epoch_mcc > 0:
                            lab_scale_non = 1 - val_epoch_mcc

                        else:
                            lab_scale_non = 1.0

                        lab_scale_vio = 1.0

                    else:

                        if val_epoch_mcc > 0:
                            lab_scale_vio = 1 - val_epoch_mcc

                        else:
                            lab_scale_vio = 1.0

                        lab_scale_non = 1.0

                    docs_class_non[train_non_len + docidx][bidx] = 1 - tree_scale_non * lab_scale_non
                    docs_class_vio[train_non_len + docidx][bidx] = 1 - tree_scale_vio * lab_scale_vio

            else:

                temp_factors_non = ADF.weightvector[0][-base_factors:]
                temp_factors_vio = ADF.weightvector[1][-base_factors:]

                for bidx in range(base_factors):

                    docs_class_non[train_non_len + docidx][bidx] = temp_factors_non[bidx]
                    docs_class_vio[train_non_len + docidx][bidx] = temp_factors_vio[bidx]
        
        print("Pos documents processed")
        
        assert docs_class_non.shape == docs_class_vio.shape == (train_non_len + train_vio_len, base_factors)

        for fidx in range(base_factors):
            
            #print("Factor ", fidx)

            input_non = []
            input_vio = []

            for didx in range(set_size):

                non_prob, vio_prob, document = gen_doc_weights(didx, fidx, train_non, train_vio, docs_class_non, docs_class_vio)
                
                if random.uniform(0, 1) <= non_prob:
                    input_non.append(copy.copy(document))

                if random.uniform(0, 1) <= vio_prob:
                    input_vio.append(copy.copy(document))
            
            del non_prob, vio_prob

            if len(input_non) != 0 and len(input_vio) != 0:
                
                comb_idx = 0

                while len(input_non) != len(input_vio):

                    didx = divmod(comb_idx, set_size)[1]

                    non_prob, vio_prob, document = gen_doc_weights(didx, fidx, train_non, train_vio, docs_class_non, docs_class_vio)

                    if len(input_non) > len(input_vio):

                        if random.uniform(0, 1) <= vio_prob:
                            input_vio.append(copy.copy(document))

                    else:

                        if random.uniform(0, 1) <= non_prob:
                            input_non.append(copy.copy(document))

                    comb_idx += 1

            ## Setting up NLP training data set.
            train_text_emb, train_label_emb = train_embeddings(input_non, input_vio, max_len)

            del input_non, input_vio
            
            #print("Embeddings processed")

            if len(train_text_emb) > 0:

                index_shuffle = shuffle([i for i in range(len(train_label_emb))])
                #print("index_shuffle: " + str(len(index_shuffle)))
    
                X_train, y_train = train_text_emb[index_shuffle], train_label_emb[index_shuffle]
                
                del train_text_emb, train_label_emb

                normalizer = Normalize()
                X_train = normalizer.normalize_train(X_train, max_len) 

                tensor_train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
                tensor_train_y = torch.from_numpy(y_train).type(torch.LongTensor)

                del X_train, y_train

                training_set = torch.utils.data.TensorDataset(tensor_train_x, tensor_train_y) # create your datset
                trainloader = torch.utils.data.DataLoader(training_set, batch_size = train_batch, shuffle = True, num_workers = 1)
		
                with open("model_%s.p"%(fidx), 'rb') as fp:
                    model = pickle.load(fp)

                model.cuda(1)

                opt = torch.optim.Adam(lr = lr, params = model.parameters())

                for i, data in enumerate(trainloader):
            
                    model.train(True)
    
                    opt.zero_grad()

                    inputs, labels = data

                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]
                
                    if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda(1)), labels.cuda(1)

                    out = model(inputs)
                    weight = [1.0, 1.0]

                    #print('balanced weight: ',weight)
                    weight = torch.tensor(weight).cuda(1)
                    loss = nn.CrossEntropyLoss(weight,reduction = 'mean')

                    output = loss(out[0], labels)

                    #print('epoch ',e,'factor ',fidx, 'step ',i,'loss:',output.item(),'num of postives', labels.sum())
                    train_loss_tol = float(output.cpu())

                    output.backward()

                    if gradient_clipping > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                    opt.step()

                    del inputs, labels, out, output
                    torch.cuda.empty_cache()

                    losses.append(train_loss_tol)          

                with torch.no_grad():
                
                    model.train(False)

                    y_train_pred = []
                    y_train_true = []

                    for idx,data in enumerate(trainloader):

                        inputs, labels = data

                        if inputs.size(1) > config.seq_length:
                            inputs = inputs[:, :config.seq_length, :]

                        if torch.cuda.is_available():
                            inputs, labels = Variable(inputs.cuda(1)), labels.cuda(1)

                        out = model(inputs)
                        
                        sm = nn.Softmax(dim = 1)
                        pred_prob = out[0].cpu()
                        pred_prob = sm(pred_prob)
                          
                        predict = torch.argmax(pred_prob, axis = 1)
                        labels = labels.cpu()
                        y_train_pred = y_train_pred + predict.tolist()
                        y_train_true = y_train_true + labels.tolist()

                        del inputs, labels, out
                    
                    train_acc = accuracy_score(y_train_true, y_train_pred)
                    train_f_score = f1_score(y_train_true, y_train_pred, average = 'macro')
                    train_tn, train_fp, train_fn, train_tp = confusion_matrix(y_train_true, y_train_pred, labels = [0, 1]).ravel()
                    
                    train_fact_mcc = matthews_corrcoef(y_train_true, y_train_pred)

                    #print('Epoch: ', e, ' factor: ', fidx, ' step: ', i, ' training accuracy: ', train_acc)
                    #print('Epoch: ', e, ' factor: ', fidx, ' step: ', i, ' training MCC: ', train_fact_mcc)
                    #print('Epoch: ', e, ' factor: ', fidx, ' step: ', i, ' training confusion matrix: ', 'TP',train_tp, 'TN',train_tn, 'FP',train_fp, 'FN',train_fn)
            
                ## Adjusting input labellings for next epoch by base-level factor.        
                with open("model_%s.p"%(fidx), 'wb') as fp:
                    pickle.dump(model, fp, pickle.HIGHEST_PROTOCOL)
            
                non_baselabs = factorascribe(train_non, model, max_len, dataset, fidx, non_baselabs, train_batch, base_factors)
                vio_baselabs = factorascribe(train_vio, model, max_len, dataset, fidx, vio_baselabs, train_batch, base_factors)
                    
                del model, opt

        del docs_class_non, docs_class_vio
        
        ## Analysis of training data
        y_train_true, y_train_pred = eval_stats(train_non_len, train_vio_len, non_baselabs, vio_baselabs)

        acc = accuracy_score(y_train_true, y_train_pred)
        f_score = f1_score(y_train_true, y_train_pred, average = 'macro')
        tn, fp, fn, tp = confusion_matrix(y_train_true, y_train_pred, labels = [0, 1]).ravel()

        train_epoch_mcc = matthews_corrcoef(y_train_true, y_train_pred)
        train_mccs.append(train_epoch_mcc)

        train_confusion_matrices.append({'TP':tp, 'TN':tn, 'FP': fp, 'FN':fn})
        
        print("Epoch: ", e, " training analatics: ")
        print(classification_report(y_train_true, y_train_pred), "acc: ", acc, "f1: ", f_score, "mcc: ", train_epoch_mcc)

        ## Analysis of test data
        test_non_baselabs = [["undec"] * ADF.depth for _ in range(test_non_len)]
        test_vio_baselabs = [["undec"] * ADF.depth for _ in range(test_vio_len)]
        
        for fidx in range(base_factors):
            
            with open("model_%s.p"%(fidx), 'rb') as fp:
                model = pickle.load(fp)
            
            model.cuda(1)

            val_non_baselabs = factorascribe(val_non, model, max_len, dataset, fidx, val_non_baselabs, eval_batch, base_factors)
            val_vio_baselabs = factorascribe(val_vio, model, max_len, dataset, fidx, val_vio_baselabs, eval_batch, base_factors)
            test_non_baselabs = factorascribe(test_non, model, max_len, dataset, fidx, test_non_baselabs, eval_batch, base_factors)
            test_vio_baselabs = factorascribe(test_vio, model, max_len, dataset, fidx, test_vio_baselabs, eval_batch, base_factors)
        
        del model

        y_val_true, y_val_pred = eval_stats(val_non_len, val_vio_len, val_non_baselabs, val_vio_baselabs)
        y_test_true, y_test_pred = eval_stats(test_non_len, test_vio_len, test_non_baselabs, test_vio_baselabs)
        
        acc = accuracy_score(y_test_true, y_test_pred)
        f_score = f1_score(y_test_true, y_test_pred, average = 'macro')
        tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred, labels = [0, 1]).ravel()
        
        ## val_epoch_mcc is used to adjust learning in the next epoch.
        val_epoch_mcc = matthews_corrcoef(y_val_true, y_val_pred)
        learn_non = np.concatenate((train_non, val_non), axis = 0)
        learn_vio = np.concatenate((train_vio, val_vio), axis = 0)
        learn_non_baselabs = non_baselabs + val_non_baselabs
        learn_vio_baselabs = vio_baselabs + val_vio_baselabs

        test_epoch_mcc = matthews_corrcoef(y_test_true, y_test_pred)
        test_mccs.append(test_epoch_mcc)
        
        test_confusion_matrices.append({'TP':tp, 'TN':tn, 'FP': fp, 'FN':fn})
        
        print("Epoch: ", e, " test analatics: ")
        print(classification_report(y_test_true, y_test_pred), "acc: ", acc, "f1: ", f_score, "mcc: ", test_epoch_mcc)
        macro_f.append(f_score)
        
        ## Recording best models
        best_models_dirs = os.path.join(args['output_dir'],'%s'%(dataset), script_name)

        if not os.path.exists(best_models_dirs):
            os.makedirs(best_models_dirs)

        ## if current best MCC score is smaller than the current epoch MCC score, we will update the attention scores
        if test_best_mcc < test_epoch_mcc:
        
            test_best_mcc = test_epoch_mcc
            
            ## Storing best models.
            for f in os.listdir(best_models_dirs):
                os.remove(os.path.join(best_models_dirs, f))
            
            for fidx in range(base_factors):
                shutil.copy("model_%s.p"%(fidx), best_models_dirs + "/" + unique_id + 'factor%s_mcc%s.pt'%(fidx, test_epoch_mcc))
            
            for fidx in range(base_factors):
                
                train_factor_ascribe[fidx] = y_train_pred
                test_factor_ascribe[fidx] = y_test_pred

    ## Recording the results.
    results_dirs = os.path.join(args['output_dir'],'%s'%(dataset), save_name, unique_id) 

    if not os.path.exists(results_dirs):
        os.makedirs(results_dirs)

    with open(os.path.join(results_dirs, "train_ascription.p"), "wb") as fp:
        pickle.dump(train_factor_ascribe, fp)

    with open(os.path.join(results_dirs, "train_confusion.p"), "wb") as fp:
        pickle.dump(train_confusion_matrices, fp)

    with open(os.path.join(results_dirs, "train_mcc.p"), "wb") as fp:
        pickle.dump(train_mccs, fp)

    with open(os.path.join(results_dirs, "test_ascription.p"), "wb") as fp:
        pickle.dump(test_factor_ascribe, fp)

    with open(os.path.join(results_dirs, "test_confusion.p"), "wb") as fp:
        pickle.dump(test_confusion_matrices, fp)

    with open(os.path.join(results_dirs, "test_mcc.p"), "wb") as fp:
        pickle.dump(test_mccs, fp)       

    for fidx in range(base_factors):
        os.remove("model_%s.p"%(fidx))






