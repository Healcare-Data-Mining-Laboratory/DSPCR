import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_cluster_new import PatientDataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from base_model_DSPCR import mllt

from transformers import AutoTokenizer
from load_label_descript import label_descript
import copy

SEED = 2022 #gpu23  model 1
SEED = 1533 #gpu23  model 2
SEED = 1099 #gpu23  model 3
# SEED = 2000 #gpu24 model 4
SEED = 2001#gpu23 model 5
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,2"

from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
class_3 = True
self_att = "self_att"
num_epochs = 30
max_length = 300
BATCH_SIZE = 25
evaluation = True
pretrained = False
Freeze = False
SV_WEIGHTS = True
Logging = False
visit = 'once'

weight_dir = "mimiciii/weights/.pth"


Best_Roc = 0.7
Best_F1 = 0.8
save_dir= "mimiciii/weights"
save_name = f""

device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0

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

def fit(epoch,model,y_bce_loss,data_length,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)

    batch_loss_list = []

    total_length = data_length

    y_list = []
    pred_list_f1 = []
    l = 0

    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()


        if flag == "train":
            with torch.set_grad_enabled(True):


                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)

                text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)

                if text['input_ids'].shape[1] > max_length:
                    text = clip_text(BATCH_SIZE,max_length,text,device)
                elif text['input_ids'].shape[1] < max_length:
                    text = padding_text(BATCH_SIZE,max_length,text,device)
                # event_list = tokenizer(event_codes_list, return_tensors="pt",padding=True).to(device)
                # print(event_codes_list)

                Yt = \
                model(text,self_att)
                loss =  y_bce_loss(Yt.squeeze(),label.squeeze())
                pred = np.array(Yt.cpu().data.tolist())

                y = np.array(label.cpu().data.tolist())
                pred=(pred > 0.5) 
                y_list.append(y)
                pred_list_f1.append(pred)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss_list.append( loss.cpu().data )  
                l+=1

        else:
            with torch.no_grad():

                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)

                text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)

                if text['input_ids'].shape[1] > max_length:
                    text = clip_text(BATCH_SIZE,max_length,text,device)
                elif text['input_ids'].shape[1] < max_length:
                    text = padding_text(BATCH_SIZE,max_length,text,device)

                # print(label_token)
                Yt = \
                model(text,self_att)
                loss =  y_bce_loss(Yt.squeeze(),label.squeeze())
                pred = np.array(Yt.cpu().data.tolist())
                y = np.array(label.cpu().data.tolist())
                pred=(pred > 0.5) 
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append(loss.cpu().data )  
                l+=1
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
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    
   
    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,total_loss))
   
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

    train_dataset = PatientDataset(f"dataset/", class_3 = class_3, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dataset/",class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model = mllt(class_3)

    if pretrained:
        print(f"loading weights: {weight_dir}")
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=True)


    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 0:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################

    y_bce_loss = nn.BCELoss()

    regularization_loss = nn.CrossEntropyLoss()

    for epoch in range(start_epoch,num_epochs):

        model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,y_bce_loss,train_length,trainloader,optimizer,flag='train')
        model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,y_bce_loss,test_length,testloader,optimizer,flag='test')



   

 







