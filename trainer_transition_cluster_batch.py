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
from transition_cluster_model import mllt


from transformers import AutoTokenizer, AutoModel

import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="1,3"

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

num_epochs = 2000
max_length = 300
attention = "self_att"

class_3 = True
cluster = True
# BATCH_SIZE = 1
BATCH_SIZE = 90
cluster_number = 10
start_epoch = 0
pretrained = True
Freeze = True
SV_WEIGHTS = True
evaluation = True
strict = False
weight_dir = ".."
center_embedding_dir = ".."


if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    strict = True
    weight_dir = ".."


# loss_ratio = [1,1e-4,1]
loss_ratio = [1,1e-4,1e-1]


Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"..."

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)

# weight_d
# weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_no_att_once_0205_epoch_1_loss_0.3046_f1_0.8739_acc_0.7326.pth"
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

    # if flag == 'train' and epoch ==0:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=True) # 这里是“欧一”，不是“零一”

    chief_comp_last = deque(maxlen=2)
    center_embedding = torch.nn.Parameter(center_embedding).to(device)

    y_list = []
    pred_list_f1 = []
    l = 0
    eopch_loss_list = []
    epoch_classify_loss_list = []
    epoch_kl_loss_list = []
    epoch_cluster_loss_list = []
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
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
                        # print(torch.tensor(label).to(torch.float32).to(device))
                        # cheif_complaint = cheif_complaint_list[d]
                        label = torch.tensor(label).to(torch.float32).to(device)
                        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)

                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)
                        if v == 0:
                            Ztd_last = Ztd_zero
                        phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,chief_comp_last,text,Ztd_last,flag,cluster,attention)

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
                        pred = (pred > 0.5) 
                        y_list.append(label)
                        pred_list_f1.append(pred)
                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = Kl_loss.view(-1).mean()
                    batch_cls_list[p] = cls_loss_p
                    batch_KLL_list[p] = kl_loss_p

                phi_batch = torch.cat(phi_list,0)

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

                        if text['input_ids'].shape[1] > max_length:
                            text = clip_text(BATCH_SIZE,max_length,text,device)
                        elif text['input_ids'].shape[1] < max_length:
                            text = padding_text(BATCH_SIZE,max_length,text,device)
                        if v == 0:
                            Ztd_last = Ztd_zero
                        phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,chief_comp_last,text,Ztd_last,flag,cluster,attention)
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
                        pred = (pred > 0.5) 
                        y_list.append(label)
                        pred_list_f1.append(pred)
                    cls_loss_p = cls_loss.view(-1).mean()
                    kl_loss_p = Kl_loss.view(-1).mean()
                    batch_cls_list[p] = cls_loss_p
                    batch_KLL_list[p] = kl_loss_p

                phi_batch = torch.cat(phi_list,0)
                # print(phi_batch[:5,:])

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

    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_f1,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_f1,average="macro")
    total_loss = sum(eopch_loss_list) / len(eopch_loss_list)
    total_cls = sum(epoch_classify_loss_list) / len(epoch_classify_loss_list)
    total_kls = sum(epoch_kl_loss_list) / len(epoch_kl_loss_list)
    total_clus = sum(epoch_cluster_loss_list) / len(epoch_cluster_loss_list)

    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} | Total Cluster LOSS  : {} ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro, total_loss,total_cls,total_kls,total_clus))
    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"xx.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"xx.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return  model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro    




if __name__ == '__main__':

    train_dataset = PatientDataset(f'xx',class_3,visit,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f'xx',class_3,visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    center_embedding = torch.load(center_embedding_dir).type(torch.cuda.FloatTensor)
    print("loading center: ",center_embedding_dir)

    print(train_dataset.__len__())
    print(test_dataset.__len__())

    model = mllt(class_3)

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=strict)
        print("loading weight: ",weight_dir)

    ### freeze parameters ####
    optimizer = optim.Adam(model.parameters(True), lr = 1e-5, weight_decay = 0.0005)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 10:
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
    cluster_loss = nn.KLDivLoss(reduction='sum')

    if evaluation:
        precision_micro_list = []
        precision_macro_list = []
        recall_micro_list = []
        recall_macro_list = []
        f1_micro_list = []
        f1_macro_list = []
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(1):
    
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,center_embedding,text_recon_loss,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')
            precision_micro_list.append(precision_micro)
            precision_macro_list.append(precision_macro)
            recall_micro_list.append(recall_micro)
            recall_macro_list.append(recall_macro)            
            f1_micro_list.append(f1_micro)
            f1_macro_list.append(f1_macro)                    
            roc_micro_list.append(roc_micro)
            roc_macro_list.append(roc_macro)
        precision_micro_mean = np.mean(precision_micro_list)
        precision_macro_mean = np.mean(precision_macro_list)        
        recall_micro_mean = np.mean(recall_micro_list)
        recall_macro_mean = np.mean(recall_macro_list)        
        f1_micro_mean = np.mean(f1_micro_list)
        f1_macro_mean = np.mean(f1_macro_list)
        roc_micro_mean = np.mean(roc_micro_list)
        roc_macro_mean = np.mean(roc_macro_list)
        print(" Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {}  ".format(precision_micro_mean,precision_macro_mean,recall_micro_mean,recall_macro_mean,f1_micro_mean,f1_macro_mean,roc_micro_mean,roc_macro_mean))

    else:
        for epoch in range(start_epoch,num_epochs):

            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,text_recon_loss,y_bce_loss,cluster_loss,trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,text_recon_loss,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')



    







