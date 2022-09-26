import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import BartTokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
import random 
from datetime import datetime
SEED = 2019
torch.manual_seed(SEED)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)
class PatientDataset(object):
    def __init__(self, data_dir,class_3,visit,flag="train",):
        self.visit = visit
        self.data_dir = data_dir
        self.flag = flag
        self.text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
        self.event_dir = '/home/comp/cssniu/mllt/dataset/event_new/'
        self.datedf = pd.read_csv('/home/comp/cssniu/mllt/dataset/new_packed_data/date_file.csv')
        self.stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
        self.class_3 = class_3

        self.sbj_dir = os.path.join(f'{data_dir}',flag)
        # self.sbj_list = sorted((os.listdir(self.sbj_dir)), key=lambda k: random.random()) 
        self.sbj_list = os.listdir(self.sbj_dir)
        self.max_length = 1000
        self.label_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def padding_text(self,vec):
        input_ids = vec['input_ids']
        attention_mask = vec['attention_mask']
        padding_input_ids = torch.ones((input_ids.shape[0],self.max_length-input_ids.shape[1]),dtype = int).to(self.device)
        padding_attention_mask = torch.zeros((attention_mask.shape[0],self.max_length-attention_mask.shape[1]),dtype = int).to(self.device)
        input_ids_pad = torch.cat([input_ids,padding_input_ids],dim=-1)
        attention_mask_pad = torch.cat([attention_mask,padding_attention_mask],dim=-1)
        vec = {'input_ids': input_ids_pad,
        'attention_mask': attention_mask_pad}
        return vec
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)

        return temp

    def __getitem__(self, idx):
        
        if self.visit == 'once':
            visit_list = [self.sbj_list[idx]]
            patient_id = self.sbj_list[idx].split("_")[0]
        else:
            patient_id = self.sbj_list[idx]
            visit_list = sorted(os.listdir(os.path.join(self.data_dir,self.flag, patient_id)), key= self.sort_key)
        # print(visit_list)

        breif_course_list = []
        label_list = []
        event_list = []
        dir_list = []
        cheif_complaint_list = []
        text_df = pd.read_csv(self.text_dir+"_".join(visit_list[0].split("_")[:2])+".csv").values
        date_init = self.datedf[self.datedf['file_name'] == "dataset/event_new/"+visit_list[0]]['date'].values[0]
        time_stamp_list = []
        for v in visit_list:
            text_df = pd.read_csv(self.text_dir+"_".join(v.split("_")[:2])+".csv").values
            date = self.datedf[self.datedf['file_name'] == "dataset/event_new/"+v]['date'].values[0]
            time_difference = datetime.fromtimestamp(datetime.strptime(date, '%Y-%m-%d').timestamp()) - datetime.fromtimestamp(datetime.strptime(date_init, '%Y-%m-%d').timestamp()) 
            time_stamp_list.append(int(time_difference.days))
            breif_course = text_df[:,1:2].tolist()
            cheif_complaint = text_df[:,0:1].tolist()
            dir_list.append(self.text_dir+"_".join(v.split("_")[:2])+".csv")
            breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
            text = ' '.join(breif_course)
            text = self.rm_stop_words(text)

            breif_course_list.append(text)
            cheif_complaint = [n[0] for n in cheif_complaint if not pd.isnull(n)]
            cheif_complaint_list.append(cheif_complaint)
   

            label = list(pd.read_csv(os.path.join(self.data_dir,self.flag,str(v.split("_")[:1][0]),v))[self.label_list].values[:1,:][0])

            cluster_label = [0,0,0]
            if self.class_3:
                if sum(label[:13]) >=1:
                    cluster_label[0] = 1
                if sum(label[13:20]) >= 1:
                    cluster_label[1] = 1
                if sum(label[20:]) >= 1:
                    cluster_label[2] = 1
                label_list.append(cluster_label)
            else:
                label_list.append(label)
            # print(cluster_label)
        return cheif_complaint_list,breif_course_list,label_list


    def __len__(self):
        return len(self.sbj_list)


def collate_fn(data):
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1] for d in data]
    label_list =[d[2] for d in data]

    return cheif_complaint_list,text_list,label_list

