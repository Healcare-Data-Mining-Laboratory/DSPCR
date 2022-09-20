import re
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
import sys,logging
from transformers import AutoTokenizer, AutoModel
from numpy.testing import assert_almost_equal
import botorch


class mllt(nn.Module):
    def __init__(self,class_3,latent_ndims):
        super(mllt, self).__init__()
        self.class_3 = class_3
        self.latent_ndims = latent_ndims
        self.hidden_size = 768
        self.alpha = 5
        self.drop_out = nn.Dropout(0.3)
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.cluster_layer = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.PReLU()
            # nn.Tanh()

            )
        self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 100),
                nn.Dropout(0.3),
                nn.Linear(100, 3),
                )
 
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.transd = nn.GRU(input_size = self.hidden_size//2, batch_first=True, hidden_size =  self.hidden_size//2, num_layers=1, bidirectional=True)

        self.prior_beta = nn.Linear(  self.hidden_size//2,  latent_ndims)

        self.transRNN =  nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size = self.hidden_size//2, num_layers=1, bidirectional=True)

     
        self.forget_gate =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            )
        self.Ztd_cat = nn.Linear(self.hidden_size, self.hidden_size//2)

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def get_cluster_prob(self, embeddings,Center):
        # print(embeddings.unsqueeze(1)[[0],:,:] - Center)

        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center) ** 2, -1)

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        # print(numerator.shape)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()


    def clustering(self,Ztd,Center):
        Center = self.cluster_layer(Center) ## 8x384

        phi = self.get_cluster_prob(Ztd,Center)#phi batch x 10 x 1
        # print("phi: ",phi[:,:])
        # print("z",Ztd[:3,:10].unsqueeze(1))
        # print("Center",Center[:5,:10])
        # print("divition: ",(Ztd[:5,:].unsqueeze(1) - Center).sum(-1)[:,:10])
        
        cluster_target =self.target_distribution(phi).detach()
        # print(phi[:1,:],cluster_target[:1,:])

        ct = self.drop_out(Center * phi.unsqueeze(-1)).sum(1)
        return ct,phi,Center
    

    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        # print("v :", v)
        # print("c :", c)

        v = self.drop_out(self.fc_key(v))
        c = self.drop_out(self.fc_query(c))
        g = torch.bmm(v, c.transpose(-2, -1))

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]

        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print("b: ",b.squeeze().squeeze(),torch.max(b),torch.sum(b))
        return b  
    
    def approximation(self,Ot,Ztd_list,flag,self_att):
              
        Ot_E_batch = self.encoder(**Ot).last_hidden_state

        # Ot_E_batch_cross_attention = 0
        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ot_E_batch_cross_attention =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
       
        else:
            Ot_E_batch_cross_attention = self.drop_out(self.fc_value(Ot_E_batch)).sum(1)
            # print(Ot_E_batch_cross_attention)
        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        # print("Ztd_last: ",Ot_E_batch_cross_attention[:80])

        Ztd_last =  self.drop_out(torch.mean(Ztd_last,0))
        Ztd = torch.cat((Ztd_last,Ot_E_batch_cross_attention),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)
        Ztd = self.drop_out( self.Ztd_cat(gate_ratio_ztd*Ztd))

        return Ztd
   
    def trasition(self,Ztd_last):
    

        _,Ztd_last_last_hidden = self.transd(Ztd_last.unsqueeze(0))

        Ztd =  self.drop_out(torch.mean(Ztd_last_last_hidden,0))
        Ztd_logvar =  self.drop_out(self.prior_beta(Ztd))

        return Ztd_logvar

    def forward(self,Ztd_list,Ztd_last,Ot,Center,flag,self_att):
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)

        Ztd = self.approximation(Ot,Ztd_list,flag,self_att)
        prior_beta = self.trasition(Ztd_last)
        u,pi,Center = self.clustering(Ztd,Center)
        y =  self.sigmoid(self.MLPs(u))
        return y,pi,prior_beta,u



        

       

    


   