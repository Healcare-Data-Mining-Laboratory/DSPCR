import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_transition import PatientDataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from Parametric_DSPCR_Single_Prior import mllt
from collections import Counter
from sklearn.metrics import *

from transformers import AutoTokenizer
from load_label_descript import label_descript
import copy
SEED = 2022 #gpu23 model 1
SEED = 1533 #gpu23 model 2
SEED = 1099 #gpu23 model 3
SEED = 2000 #gpu23 model 4
SEED = 2001 #gpu23 model 5
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="1,3"

from transformers import AutoTokenizer, AutoModel

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
train_base = False
class_3 = True
self_att = "self_att"
loss_ratio = [1,1e-1,1e-4]
num_epochs = 200
max_length = 300
BATCH_SIZE = 10
latent_ndims = 8
# prior_beta_0 = concentration_alpha0 = torch.Tensor([5.])

evaluation = False
pretrained = True
Freeze = True
SV_WEIGHTS = True


Best_Roc = 0.7
Best_F1 = 0.8
visit = 'twice'
save_dir= "mimiciii/weights/"
save_name = f"para_DSTFE_single_prior_K{latent_ndims}_{self_att}_{visit}_0920_{SEED}"
if Logging:
    logging_text = open(f"{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0

weight_dir = "mimiciii/weights/.pth"

if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    weight_dir = "mimiciii/weights/.pth"



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


def fit(epoch,model,center_embedding,y_bce_loss,trans_loss,cluster_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc,prior_alpha
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    trans_loss.to(device)
    cluster_loss.to(device)
    center_embedding = torch.nn.Parameter(center_embedding).to(device)

    eopch_loss_list = []
    epoch_classify_loss_list = []
    epoch_kl_loss_list = []
    epoch_cluster_loss_list = []
    cluster_id_list = []
    embedding_list = []
    cluster_id_list = []

    labels_list = []
    y_list_f1 = []
 
    l = 0

    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        batch_cls_list = torch.zeros(len(text_list)).to(device)
        batch_tansl_list = torch.zeros(len(text_list)).to(device)

        pi_list = []

      
        if flag == "train":
            with torch.set_grad_enabled(True):
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_label = label_list[p]
                    Ztd_zero = torch.randn((1, 384)).to(device)
                    Ztd_zero.requires_grad = True
                    sbkl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)
                    tans_loss = torch.zeros(len(p_text)).to(device)
            
                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]

                    Prior_beta_last =  torch.zeros((1)).to(device)
                    Prior_beta_last.requires_grad = True

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
                        ## mean alpha logvar beta
                        y,pi,prior_beta,u  = \
                        model(Ztd_list,Ztd_last,text,center_embedding,flag,self_att)
                        pi_list.append(pi)
                        embedding_list.append(u.squeeze().cpu().data.tolist())

                        Ztd_last = u 
                        Ztd_list.append(u)
                        s_cls =  y_bce_loss(y.squeeze(),label.squeeze())
                        s_transl = trans_loss(Prior_beta_last,prior_beta)
                        Prior_beta_last = prior_beta
                        cls_loss[v] = s_cls
                        tans_loss[v] = s_transl

                        label = np.array(label.cpu().data.tolist())
                        y = np.array(y.cpu().data.tolist())
                        y = (y > 0.5) 

                        labels_list.append(label)
                        y_list_f1.append(y)

                    cls_loss_p = cls_loss.view(-1).mean()
                    tans_loss_p = tans_loss.view(-1).mean()

                    batch_cls_list[p] = cls_loss_p
                    batch_tansl_list[p] = tans_loss_p

                pi_batch = torch.cat(pi_list,0)
            
                cluster_id = torch.argmax(pi_batch,dim = -1)
                for i in cluster_id:
                    cluster_id_list.append(i.cpu().data.tolist())
                target_t  = model.target_distribution(pi_batch).detach()
                batch_cluster_lss = cluster_loss((pi_batch+1e-08).log(),target_t)/pi_batch.shape[0]
                batch_cls_list = batch_cls_list.view(-1).mean()
                batch_tansl_list = batch_tansl_list.view(-1).mean()
                total_loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_cluster_lss + loss_ratio[2]*batch_tansl_list
                total_loss.backward(retain_graph=True)
                optimizer.step()

                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_kl_loss_list.append(batch_tansl_list.cpu().data) 
                epoch_cluster_loss_list.append(batch_cluster_lss.cpu().data)

        else:
            with torch.no_grad():
                for p in range(len(text_list)):
                    p_text = text_list[p]
                    p_label = label_list[p]
                    Ztd_zero = torch.randn((1, 384)).to(device)
                    Ztd_zero.requires_grad = True
                    sbkl_loss = torch.zeros(len(p_text)).to(device)
                    cls_loss = torch.zeros(len(p_text)).to(device)
                    tans_loss = torch.zeros(len(p_text)).to(device)
                    Ztd_last = Ztd_zero
                    Ztd_list = [Ztd_zero]

                    Prior_beta_last =  torch.zeros((1)).to(device)
                    Prior_beta_last.requires_grad = True

                    label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)
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
                        ## mean alpha logvar beta
                                               y,pi,prior_beta,u  = \
                        model(Ztd_list,Ztd_last,text,center_embedding,flag,self_att)
                        pi_list.append(pi)
                        embedding_list.append(u.squeeze().cpu().data.tolist())

                        Ztd_last = u 
                        Ztd_list.append(u)
                        s_cls =  y_bce_loss(y.squeeze(),label.squeeze())
                        s_transl = trans_loss(Prior_beta_last,prior_beta)
                        Prior_beta_last = prior_beta
                        cls_loss[v] = s_cls
                        tans_loss[v] = s_transl

                        label = np.array(label.cpu().data.tolist())
                        y = np.array(y.cpu().data.tolist())
                        y = (y > 0.5) 

                        labels_list.append(label)
                        y_list_f1.append(y)

                    cls_loss_p = cls_loss.view(-1).mean()
                    tans_loss_p = tans_loss.view(-1).mean()

                    batch_cls_list[p] = cls_loss_p
                    batch_tansl_list[p] = tans_loss_p

                pi_batch = torch.cat(pi_list,0)
            
                cluster_id = torch.argmax(pi_batch,dim = -1)
                for i in cluster_id:
                    cluster_id_list.append(i.cpu().data.tolist())
                batch_cluster_lss = cluster_loss((pi_batch+1e-08).log(),target_t)/pi_batch.shape[0]
                batch_cls_list = batch_cls_list.view(-1).mean()
                batch_tansl_list = batch_tansl_list.view(-1).mean()
                total_loss = loss_ratio[0]*batch_cls_list + loss_ratio[1]*batch_cluster_lss + loss_ratio[2]*batch_tansl_list

                eopch_loss_list.append(total_loss.cpu().data )  
                epoch_classify_loss_list.append(batch_cls_list.cpu().data) 
                epoch_kl_loss_list.append(batch_tansl_list.cpu().data) 
                epoch_cluster_loss_list.append(batch_cluster_lss.cpu().data)

    label_count = Counter(cluster_id_list).most_common()
    labels_list = np.vstack(labels_list)
    y_list_f1 = np.vstack(y_list_f1)
    cluster_id_list = np.array(cluster_id_list)
    embedding_list =  np.array(embedding_list)
    shi_score = silhouette_score(embedding_list, cluster_id_list)
    db_score = davies_bouldin_score(embedding_list, cluster_id_list)
    vic_score = calinski_harabasz_score(embedding_list, cluster_id_list)

    precision_micro = metrics.precision_score(labels_list,y_list_f1,average='micro')
    recall_micro =  metrics.recall_score(labels_list,y_list_f1,average='micro')
    precision_macro = metrics.precision_score(labels_list,y_list_f1,average='macro')
    recall_macro =  metrics.recall_score(labels_list,y_list_f1,average='macro')

    f1_micro = metrics.f1_score(labels_list,y_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(labels_list,y_list_f1,average="micro")
    f1_macro = metrics.f1_score(labels_list,y_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(labels_list,y_list_f1,average="macro")
    total_loss = sum(eopch_loss_list) / len(eopch_loss_list)
    total_cls_loss = sum(epoch_classify_loss_list) / len(epoch_classify_loss_list)
    total_trans_kl_loss = sum(epoch_kl_loss_list) / len(epoch_kl_loss_list)
    total_cluster_loss = sum(epoch_cluster_loss_list) / len(epoch_cluster_loss_list)

    print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC: {} |  SHI Score: {} |  DB Score: {} | VIC Score: {} | CLS LOSS  : {} | Trans KL LOSS  : {} | Cluster LOSS  : {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,shi_score,db_score,vic_score,total_cls_loss,total_trans_kl_loss,total_cluster_loss,total_loss))

    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f".pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f".pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)

    return model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro

if __name__ == '__main__':
    
    print("load center from ", ".pth")
    center_embedding = torch.load("dataset/.pth").type(torch.cuda.FloatTensor)


    train_dataset = PatientDataset(f"dataset/", class_3 = class_3, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dataset/",class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model = mllt(class_3,latent_ndims)

    if pretrained:
        print(f"loading weights: {weight_dir}")
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)


    # optimizer = optim.Adam(model.parameters(True), lr = 1e-5)
    ignored_params = list(map(id, model.encoder.parameters())) 

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters()) 
    optimizer = optim.Adam([
    {'params': base_params},
    {'params': model.encoder.parameters(), 'lr': 1e-5}], 3e-4)

        for (i,child) in enumerate(model.children()):
            if i == 6:
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False
    ##########################


    y_bce_loss = nn.BCELoss()
    trans_loss = nn.MSELoss()
    cluster_loss = nn.KLDivLoss(reduction='sum')


    for epoch in range(start_epoch,num_epochs):

        model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,y_bce_loss,trans_loss,cluster_loss,trainloader,optimizer,flag='train')
        model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,y_bce_loss,trans_loss,cluster_loss,testloader,optimizer,flag='test')





 







