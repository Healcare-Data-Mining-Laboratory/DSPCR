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


class mllt(nn.Module):
    def __init__(self,class_3):
        super(mllt, self).__init__()
        self.class_3 = class_3
        self.alpha = 10
        self.hidden_size = 768
        self.last_patient_id = deque([0])
        self.drop_out = nn.Dropout(0.3)

        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.cluster_layer = nn.Sequential(
            nn.Linear(self.hidden_size//2, self.hidden_size//2),
            nn.Dropout(0.3),
            nn.PReLU()
            )
        if class_3:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 3),
                )
        else:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 25),
                )
        self.transd = nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size= self.hidden_size//2, num_layers=1, bidirectional=True)

        self.transd_mean = nn.Linear( self.hidden_size//2,  self.hidden_size//2)
        self.transd_logvar = nn.Linear( self.hidden_size//2,  self.hidden_size//2)

        self.transRNN =  nn.GRU(input_size= self.hidden_size//2, batch_first=True, hidden_size= self.hidden_size//2, num_layers=1, bidirectional=True)

        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.zd_mean = nn.Linear( self.hidden_size//2,  self.hidden_size//2)
        self.zd_logvar = nn.Linear( self.hidden_size//2,  self.hidden_size//2)  
        self.Ztd_cat = nn.Linear(self.hidden_size, self.hidden_size//2)

      
        self.forget_gate =  nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            )


        self.fusion_fc = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.BN = nn.BatchNorm1d(8)

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
   

    

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
        

    def sampling(self,mu,logvar,flag):
        if flag == "test":
            return mu
        std = torch.exp(0.5 * logvar).detach()        
        epsilon = torch.randn_like(std).detach()
        zt = epsilon * std + mu 
        return zt
        
    def get_cluster_prob(self, embeddings,Center):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center)**2, -1)
        # norm_squared =  F.pairwise_distance(embeddings.unsqueeze(-1), Center.T, p=2)
        # print("norm: ",numerator[:5,:10])

        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))

        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        # print(numerator.shape)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        # print("batch: ", batch)

        # print("batch** 2: ",batch ** 2)
        # print("(torch.sum(batch, 0) + 1e-9", (torch.sum(batch, 0) + 1e-9))
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()

    def approximation(self,Ztd_list, Ot,flag,self_att):
          
        Ot_E_batch = self.encoder(**Ot).last_hidden_state

        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ot_E_batch_cross_attention =   self.drop_out(Ot_E_batch * attention_weights).sum(1)


        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        Ztd_last =  self.drop_out(torch.mean(Ztd_last,0))
        Ztd = torch.cat((Ztd_last,Ot_E_batch_cross_attention),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)
        # print(gate_ratio_ztd.shape,Ztd.shape)
        Ztd = self.drop_out( self.Ztd_cat(gate_ratio_ztd*Ztd))
        Ztd_mean = self.zd_mean(Ztd)
        Ztd_logvar = self.zd_logvar(Ztd)

        Ztd_s = self.sampling(Ztd_mean,Ztd_logvar,flag)
      
        return Ztd_s,Ztd_mean,Ztd_logvar,attention_weights
 
    def emission(self,Ztd):

        Yt =  self.sigmoid(self.MLPs(Ztd))

        return Yt

    def trasition(self,Ztd_last,chief_comp_last):


        _,Ztd_last_last_hidden = self.transd(Ztd_last.unsqueeze(0))

        Ztd =  self.drop_out(torch.mean(Ztd_last_last_hidden,0))

        Ztd_mean =   self.drop_out(self.transd_mean(Ztd))
        Ztd_logvar =  self.drop_out(self.transd_logvar(Ztd))

        return Ztd_mean,Ztd_logvar

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

    def forward(self,Center,Ztd_list,Ct,Ot,Ztd_last,flag,cluster,self_att):
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)
        Ztd,Ztd_mean_post,Ztd_logvar_post,attention_weights = self.approximation(Ztd_list, Ot,flag,self_att)
        if cluster:
            Ztd_mean_priori,Ztd_logvar_priori = self.trasition(Ztd_last,Ct)
        
            Ct,phi,Center = self.clustering(Ztd,Center)
        
            Yt = self.emission(Ct)
            return phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,Yt,Ztd_mean_priori,Ztd_logvar_priori,attention_weights

        else:
            Ztd = self.approximation(Ztd_list, Ot,flag,self_att)
            Yt = self.emission(Ztd)

            return Yt
        # return  Ot_,Yt,Ot
           

   

   