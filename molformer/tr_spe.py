""" reference from Harvard NLP: https://nlp.seas.harvard.edu/2018/04/03/attention.html"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


#######################################
## Transformer with only Encoder
#######################################

class Transformer3D(nn.Module):
    def __init__(self, encoder, src_embed, src_pe, generator):
        super(Transformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pe = src_pe
        self.generator = generator

    def forward(self, src, src_mask, pos):
        return self.generator(self.encoder(self.src_pe(self.src_embed(src), pos), src_mask)[:, 0])


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#######################################
## Encoder part
#######################################

class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """MultiRelationEncoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class LayerNorm(nn.Module):
    """ layernorm layer """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        return self.a_2 * (x - mean) / std + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm. For code simplicity the norm is first as opposed to last."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


#######################################
## attention part
#######################################

def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        self.h = h
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.d_k = embed_dim // h

    def forward(self, query, key, value, mask=None):
        # expand to H heads
        if mask is not None: mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from embed_dim => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


#######################################
## Position-wise FFN
#######################################

class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


#######################################
## Embedding part
#######################################

class Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # scale the values of token embedding
        return self.embed(x) * math.sqrt(self.embed_dim)


#######################################
## Position Encoding
#######################################

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


#######################################
## Predictor
#######################################

class Generator3D(nn.Module):
    def __init__(self, embed_dim, tgt, dropout, mean=False, binary=False):
        super(Generator3D, self).__init__()
        self.tgt = tgt
        self.mean = mean
        self.binary = binary
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(embed_dim, tgt))

    def forward(self, x, mask=None):
        if self.mean:           # protein-level prediction
            return self.proj(torch.mean(x * mask.unsqueeze(-1), dim=1)).squeeze(-1)
        res = self.proj(x)
        if self.binary:
            res = torch.sigmoid(res)
        if self.tgt == 1:       # residue-level prediction with scalar-output
            res = res.squeeze(-1)
        return res
