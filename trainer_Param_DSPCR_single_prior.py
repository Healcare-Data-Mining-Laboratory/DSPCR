from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_transition import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from transition_cluster_model import mllt
from dspcr import mllt
from sklearn.metrics.cluster import normalized_mutual_info_score

from load_label_descript import label_descript
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import *

import copy

SEED = 1533 
SEED = 1099 
SEED = 2000 
SEED = 2001 
SEED = 2019 

torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,3"

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

num_epochs = 200
max_length = 300
attention = "self_att"

class_3 = True
cluster = True
# BATCH_SIZE = 1
BATCH_SIZE = 10
cluster_number = 8
start_epoch = 0
pretrained = True
Freeze = True
SV_WEIGHTS = True
evaluation = False
strict = False



weight_dir = "xx.pth"

center_embedding_dir = "xx.pth"

if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    strict = True

    weight_dir = "xx.pth"


loss_ratio = [1,1e-4,1]


Best_Roc = 0.7
Best_F1 = 0.8
visit = 'twice'
save_dir= ".."
save_name = f".."

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)


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
    sentence_difference = max_length - len(input_ids[0])
    padding_ids = torch.ones((1,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((1,sentence_difference), dtype = torch.long).to(device)

    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec

def collate_fn(data):
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1] for d in data]
    label_list =[d[2] for d in data]

    return cheif_complaint_list,text_list,label_list


def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD



def fit(epoch,model,center_embedding,y_bce_loss,cluster_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    cluster_loss.to(device)

    y_list = []
    pred_list_f1 = []
    pred_list_roc = []
    l = 0
    eopch_loss_list = []
    epoch_classify_loss_list = []
    epoch_kl_loss_list = []
    epoch_cluster_loss_list = []
    cluster_id_list = []
    embedding_list = []
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        # if i == 10:break
        optimizer.zero_grad()
        batch_KLL_list = torch.zeros(len(text_list)).to(device)
        batch_cls_list = torch.zeros(len(text_list)).to(device)
        phi_list = []
        if flag == "train":
            with torch.set_grad_enabled(True):
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_label = label_list[p]
                    Ztd_zero = torch.randn((1, model.hidden_size//2)).to(device)
                    Ztd_zero.requires_grad = True
                    Kl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)

                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]
                
                    for v in range(len(p_text)):
                    
                        text = p_text[v]
                        label = p_label[v]
 
                        label = torch.tensor(label).to(torch.float32).to(device)
                        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
        
                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)
                        if v == 0:
                            Ztd_last = Ztd_zero
                        phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,text,Ztd_last,flag)
                        embedding_list.append(Ztd_mean_post.squeeze().cpu().data.tolist())

                        Ztd_last = Ztd_mean_post
                        Ztd_list.append(Ztd_last)
                        phi_list.append(phi)
                        icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                        if v == 0:
                            q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                        else:
                            q_ztd = KL_loss(Ztd_mean_priori,Ztd_logvar_priori,Ztd_mean_post, Ztd_logvar_post)

                        Kl_loss[v] = q_ztd
                        cls_loss[v] = icd_L
                        label = np.array(label.cpu().data.tolist())
                        pred = np.array(pred.cpu().data.tolist())
                        pred_list_roc.append(pred)

                        pred = (pred > 0.5) 
                        y_list.append(label)
                        pred_list_f1.append(pred)
                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = Kl_loss.view(-1).mean()
                    batch_cls_list[p] = cls_loss_p
                    batch_KLL_list[p] = kl_loss_p

                phi_batch = torch.cat(phi_list,0)
                cluster_id = torch.argmax(phi_batch,dim = -1)
                for i in cluster_id:
                    cluster_id_list.append(i.cpu().data.tolist())
                target_t  = model.target_distribution(phi_batch).detach()
                batch_cluster_lss = cluster_loss((phi_batch+1e-08).log(),target_t)/phi_batch.shape[0]
                batch_KLL_list = batch_KLL_list.view(-1).mean()
                batch_cls_list = batch_cls_list.view(-1).mean()
                total_loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_KLL_list + loss_ratio[2]*batch_cluster_lss
                total_loss.backward(retain_graph=True)
                optimizer.step()
                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_kl_loss_list.append(batch_KLL_list.cpu().data) 
                epoch_cluster_loss_list.append(batch_cluster_lss.cpu().data)
        else:
            with torch.no_grad():
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_label = label_list[p]
                    Ztd_zero = torch.randn((1, model.hidden_size//2)).to(device)
                    Ztd_zero.requires_grad = True
                    Kl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)

                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]
                
                    for v in range(len(p_text)):
                    
                        text = p_text[v]
                        label = p_label[v]
                        # cheif_complaint = cheif_complaint_list[d]
                        label = torch.tensor(label).to(torch.float32).to(device)
                        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        # cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                        # chief_comp_last.append(cheif_complaint)
                        # print(text)
                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)
                        if v == 0:
                            Ztd_last = Ztd_zero
                        phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,text,Ztd_last,flag)
                        embedding_list.append(Ztd_mean_post.squeeze().cpu().data.tolist())

                        Ztd_last = Ztd_mean_post
                        Ztd_list.append(Ztd_last)
                        phi_list.append(phi)
                        icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                        if v == 0:
                            q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                        else:
                            q_ztd = KL_loss(Ztd_mean_priori,Ztd_logvar_priori,Ztd_mean_post, Ztd_logvar_post)
                        Kl_loss[v] = q_ztd.cpu().data.tolist()
                        cls_loss[v] = icd_L.cpu().data.tolist()
                        label = np.array(label.cpu().data.tolist())
                        pred = np.array(pred.cpu().data.tolist())
                        pred_list_roc.append(pred)

                        pred = (pred > 0.5) 
                        y_list.append(label)
                        pred_list_f1.append(pred)
                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = Kl_loss.view(-1).mean()
                    batch_cls_list[p] = cls_loss_p
                    batch_KLL_list[p] = kl_loss_p

                phi_batch = torch.cat(phi_list,0)
                cluster_id = torch.argmax(phi_batch,dim = -1)
                for i in cluster_id:
                    cluster_id_list.append(i.cpu().data.tolist())
                target_t  = model.target_distribution(phi_batch).detach()
                batch_cluster_lss = cluster_loss((phi_batch+1e-08).log(),target_t)/phi_batch.shape[0]
                batch_KLL_list = batch_KLL_list.view(-1).mean()
                batch_cls_list = batch_cls_list.view(-1).mean()
                total_loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_KLL_list + loss_ratio[2]*batch_cluster_lss
                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_kl_loss_list.append(batch_KLL_list.cpu().data) 
                epoch_cluster_loss_list.append(batch_cluster_lss.cpu().data)
 
    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)    
    pred_list_roc = np.vstack(pred_list_roc)
    acc = metrics.accuracy_score(y_list,pred_list_f1)

    nmi = normalized_mutual_info_score(pred_list_f1.reshape(-1, 1).squeeze(), y_list.reshape(-1, 1).squeeze())
    cluster_id_list = np.array(cluster_id_list)
    embedding_list =  np.array(embedding_list)

    shi_score = silhouette_score(embedding_list, cluster_id_list)
    db_score = davies_bouldin_score(embedding_list, cluster_id_list)
    vic_score = calinski_harabasz_score(embedding_list, cluster_id_list)

    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_roc,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_roc,average="macro")

    total_loss = sum(eopch_loss_list) / len(eopch_loss_list)
    total_cls = sum(epoch_classify_loss_list) / len(epoch_classify_loss_list)
    total_kls = sum(epoch_kl_loss_list) / len(epoch_kl_loss_list)
    total_clus = sum(epoch_cluster_loss_list) / len(epoch_cluster_loss_list)

    print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC : {} | ACC: {} |  SHI Score: {} |  DB Score: {} | VIC Score: {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} | Total Cluster LOSS  : {} ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro, acc, shi_score,db_score,vic_score,total_loss,total_cls,total_kls,total_clus))
    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return  model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro    




if __name__ == '__main__':

    train_dataset = PatientDataset(f'xx/{visit}/',class_3,visit,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f'xx/{visit}/',class_3,visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    center_embedding = torch.load(center_embedding_dir).type(torch.cuda.FloatTensor)
    print("loading center: ",center_embedding_dir)

    print(train_dataset.__len__())
    print(test_dataset.__len__())

    model = mllt(class_3)

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=strict)
        print("loading weight: ",weight_dir)

    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 10:
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    y_bce_loss = nn.BCELoss()
    cluster_loss = nn.KLDivLoss(reduction='sum')

    if evaluation:

        for epoch in range(1):
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,center_embedding,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')

    else:
        for epoch in range(start_epoch,num_epochs):

            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,y_bce_loss,cluster_loss,trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')



    







