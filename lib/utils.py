import math
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg
from torch.nn.utils.weight_norm import weight_norm


def dynamic_sample(logits, p, low, high):
    # filtering
    probs = F.softmax(logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True) # [b*5, vocab_size]
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    filtered_words = cum_sum_probs < p # [b*5, vocab_size]
    filtered_words = torch.cat([filtered_words.new_ones(filtered_words.shape[0], 1), filtered_words[:, :-1]], dim=-1) # right shift
    # print(p, low, high)
    if low >-1 and high >-1: 
        # print(p, low, high)
        filtered_words[:, :low] = True
        filtered_words[:, high:] = False 

    # re-weighting
    sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
    sorted_logits[~filtered_words] = -torch.inf
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    
    # sampling
    sampled_sorted_indexes = torch.multinomial(sorted_probs, num_samples=1) # [b*5, 1]
    wt = indices.gather(1, sampled_sorted_indexes)
    logP_t = torch.log(sorted_probs.gather(1, sampled_sorted_indexes))
    
    return wt, logP_t


# def dynamic_sample(logits, p, low, high):
#     #　logits [b*5, vocab_size]

#     # filtering
#     sorted_logits, sorted_indexes = torch.sort(logits, dim=-1, descending=True) 
#     sorted_probs = F.softmax(sorted_logits, dim=-1)
#     cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
#     filtered_words = cum_sum_probs < p # [b*5, vocab_size]
#     filtered_words = torch.cat([filtered_words.new_ones(filtered_words.shape[0], 1), filtered_words[:, :-1]], dim=-1) # right shift
    
#     # print(p, low, high)
#     if low >-1 and high >-1: 
#         # print(p, low, high)
#         filtered_words[:, :low] = True
#         filtered_words[:, high:] = False 

#     # re-weighting
#     sorted_logits[~filtered_words] = -torch.inf
#     sorted_probs = F.softmax(sorted_logits, dim=-1)
    
#     # sampling
#     sampled_sorted_indexes = torch.multinomial(sorted_probs, num_samples=1) # [b*5, 1]
#     wt = sorted_indexes.gather(1, sampled_sorted_indexes)
#     logP_t = torch.log(sorted_probs.gather(1, sampled_sorted_indexes))
    
#     return wt, logP_t    




def normal_sample(logits):
    logprobs_t = F.log_softmax(logits, dim=-1)
    probs_t = torch.exp(logprobs_t) # [b*5, vocab_size]
    wt = torch.multinomial(probs_t, num_samples=1) # [b*5, 1] 
    logP_t = logprobs_t.gather(1, wt)
    return wt, logP_t

def activation(act):
    if act == 'RELU':
        return nn.ReLU(inplace=True)
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'CELU':
        return nn.CELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()

def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
    return tensor

def expand_numpy(x, size=cfg.DATA_LOADER.SEQ_PER_IMG):
    if cfg.DATA_LOADER.SEQ_PER_IMG == 1:
        return x
    x = x.reshape((-1, 1))
    x = np.repeat(x, size, axis=1)
    x = x.reshape((-1))
    return x

def load_ids(path):
    with open(path, 'r') as fid:
        lines = [int(line.strip()) for line in fid]
    return lines

def load_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines

def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab

# torch.nn.utils.clip_grad_norm
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L84-L91
# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
def clip_gradient(optimizer, model, grad_clip_type, grad_clip):
    if grad_clip_type == 'Clamp':
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad == True:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    elif grad_clip_type == 'Norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif grad_clip_type == 'None':
        pass
    else:
        raise NotImplementedError

def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float(-1e9)).type_as(t)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count