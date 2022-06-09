import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_cluster_new import PatientDataset

import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from embedding_generator_icdm import mllt

from transformers import AutoTokenizer
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0"
from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
self_att = "self_att"
num_epochs = 1
max_length = 300
BATCH_SIZE = 300
number_cluster = 10
visit = 'once'

device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)

start_epoch = 0


weight_dir = "xxh"


def get_kmeans_centers(all_embeddings, num_classes):
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    return clustering_model.cluster_centers_

def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]
    seq_mask = attention_mask[:,[-1]]
    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - input_ids.shape[1]
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)
    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)

    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec



def collate_fn(data):
    
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1][0] for d in data]
    label_list = [d[2] for d in data]
    return cheif_complaint_list,text_list,label_list


def fit(epoch,model,label_embedding,dataloader,flag='train'):
    device = device1
    model.eval()
    model.to(device)
    embed_list = []
    y_list = []
    pred_list_f1 = []
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):    
  

        with torch.no_grad():
            label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)

            text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
            label_token =  tokenizer(label_embedding, return_tensors="pt",padding=True,max_length = max_length).to(device)

            if text['input_ids'].shape[1] > max_length:
                text = clip_text(BATCH_SIZE,max_length,text,device)
            elif text['input_ids'].shape[1] < max_length:
                text = padding_text(BATCH_SIZE,max_length,text,device)
            embed = model(text,label_token,self_att)
            embed_list.append(embed)
    
    embed_list = torch.cat(embed_list,0).cpu()
    torch.save(embed_list, "xx")
    cluster_centers = torch.tensor(get_kmeans_centers(embed_list,number_cluster))

    torch.save(cluster_centers, f"xx.pth")

    


if __name__ == '__main__':
    eval_visit = "four"
    label_embedding = label_descript()

    train_dataset = PatientDataset(f"xx", class_3 = True, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = False)

    train_length = train_dataset.__len__()

    print(train_length)

    model = mllt()

    print(f"loading weights: {weight_dir}")
    model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device1)), strict=False)


    fit(1,model,label_embedding,trainloader,flag='test')



   

 







