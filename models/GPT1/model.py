"""
base code is https://github.com/paul-hyun/transformer-evolution/blob/master/gpt/model.py
"""


import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from time import time

""" sinusoid position encoding"""
def get_sinusoid_encoding_table(n_seq,d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posiangle_vec(position):
        return [cal_angle(position,i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posiangle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2]) # even index sin
    sinusoid_table[:,1::2] = np.cos(sinusoid_table[:,1::2]) # odd index cos

    return sinusoid_table

"""attention pad mask"""
def get_attn_pad_mask(seq_q,seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size,len_q,len_k)
    return pad_attn_mask

"""attention decoder mask"""
def get_attn_decoder_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of matrix(2-D)
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self,config):
        self.config = config
        self.dropout = nn.Dropout(config.attn_drop)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self,q,k,v,attn_mask):
        # (batch size, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(q,k.transpose(-2,-1)).mul_(self.scale)
        scores.masked_fill_(attn_mask,-1e9)

        # (batch size, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, v)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob
    
"""masked multi head attention"""
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        self.config = config

        self.w_q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.w_k = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.w_v = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.proj = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.proj_drop = nn.Dropout(config.proj_drop)

    def forward(self,q,k,v,attn_mask):
        B = q.size(0)
        # q shape is (batch size,n_q_seq,d_head)
        q = self.w_q(q).view(B,-1, self.config.n_head, self.config.d_head).permute(0,2,1,3) # permute(0,2,1,3) equal transpose(1,2)
        # k shape is (batch size,n_q_seq,d_head)
        k = self.w_k(k).view(B,-1, self.config.n_head, self.config.d_head).permute(0,2,1,3) # permute(0,2,1,3) equal transpose(1,2)
        # v shape is (batch size,n_q_seq,d_head)
        v = self.w_v(v).view(B,-1, self.config.n_head, self.config.d_head).permute(0,2,1,3) # permute(0,2,1,3) equal transpose(1,2)

        # (batch size, n_head, n_q_seq,n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head,1,1)

        # (batch size, n_head, n_q_seq,d_head), (batch size, n_head, n_q_seq,n_k_seq)
        context = context, attn_prob = self.scaled_dot_attn(q,k,v,attn_mask)
        # (batch size, n_head, n_q_seq, n_k_seq)
        context = context.transpose(1,2).contiguous().view(B,-1,self.config.n_head * self.confg.d_head)
        output = self.proj(context)
        output = self.proj_drop(output)
        return output, attn_prob

""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.act = F.gelu
        self.proj_drop = nn.Dropout(config.proj_drop)
    
    def  forward(self,inputs):
        # (batch size, d_ff, n_seq)
        output = self.act(self.conv1(inputs.transpose(1,2)))
        # (batch, n_seq, d_hidn)
        output = self.conv2(output).transpose(1,2)
        output = self.proj_drop(output)
        # (batch size,n_seq, d_hidn)
        return output

"""decoder layer"""
class DecoderLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self,dec_inputs,self_attn_mask):
        # (batch size, n_dec_seq, d_hidn), (batch size, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, self._attn_mask)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (batch size, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(self_att_outputs)
        ffn_outputs = self.layer_norm3(self_att_outputs + ffn_outputs)
        # (batch size, n_dec_seq, d_hidn), (batch size, n_head, n_dec_seq, n_dec_seq)
        return ffn_outputs, self_attn_prob

"""decoder"""
class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab,self.config.d_hidn)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.confg.d_hidn))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self,dec_inputs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device,dtype=dec_inputs.dtype).expand(dec_inputs.size(0),dec_inputs.size(1)).contiguous() + 1
        pos_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask,0)

        # (batch size, n_dec_seq, d_hidn)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions)

        # (batch_size, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs,dec_inputs, self.config.i_pad)

        # (batch_size, n_dec_seq, n_dec_seq)
        dec_attn_decoder_mask = get_attn_decoder_mask(dec_inputs)

        # (batch size, n_dec_seq, n_dec_seq)

        self._attn_probs = []
        for layer in self.layers:
            # (batch size, n_dec_seq, d_hidn), (batch size, n_dec_seq, n_dec_seq)
            dec_outputs, self_attn_probs = layer(dec_outputs, dec_self_attn_mask)
            self_attn_probs.append(self_attn_probs)
        # (batch size, n_dec_seq, d_hidn), [(batch size, n_dec_seq, n_dec_seq)]
        return dec_outputs, self_attn_probs

""" gpt """
class GPT(PreTrainedModel,nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

    def forward(self,dec_inputs):
        dec_outputs, dec_self_attn_probs = self.decoder(dec_inputs)
        return dec_outputs, dec_self_attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]
        