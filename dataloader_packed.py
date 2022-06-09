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
from tqdm import tqdm
from nltk.corpus import stopwords

class PatientDataset(object):
    def __init__(self, data_dir,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.lab_list = sorted(os.listdir(self.data_dir))
        self.text_dir = os.path.join(data_dir,'text',flag)
        self.text_list = sorted(os.listdir( self.text_dir))
        self.lab_dir = os.path.join('xx',flag)
        self.event_dir = os.path.join('xx',flag)
        self.all_feature_list = ['Capillary refill rate', 'Diastolic blood pressure',
       'Fraction inspired oxygen', 'Glascow coma scale eye opening',
       'Glascow coma scale motor response', 'Glascow coma scale total',
       'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height',
       'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate',
       'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
        self.max_length = 1000
        self.icd_list = ['428.0', '401.9', '38.93', '427.31', '584.9', '250.00', '414.01', '272.4', '518.81', '599.0', '530.81', '96.04', '39.95', '403.90', '585.6', '99.04', '486', '585.9', '285.9', 'V58.61', '995.92', '96.6', '403.91', '96.71', '496', '244.9', '311', '276.2', '327.23', 'V58.67', 'V45.81', '038.9', '272.0', '412', '285.21', '305.1', '785.52', '285.1', 'V45.82', '416.8', '276.7', '96.72', '276.1', 'V12.51', '507.0', '38.91', '274.9', '357.2', '287.5', '427.89']
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

        return ''.join([i for i in data.lower() if not i.isdigit()])
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
        id_ = int(re.split(r'(\d+)', text.split("_")[1])[1])
        temp.append(id_)

        return temp
    def __getitem__(self, idx):
        patient_id = self.text_list[idx]
        text_patient_id_dir = os.path.join(self.data_dir,'text',self.flag, patient_id)
        event_patient_id_dir = os.path.join(self.data_dir,'event',self.flag, patient_id)
        visit_list = sorted(os.listdir(text_patient_id_dir), key= self.sort_key)

        # visit_list = os.listdir(text_patient_id_dir)
        text_list = []
        label_list = []
        event_list = []
        print(visit_list)
        for v in visit_list:
            text_file = pd.read_csv(os.path.join(text_patient_id_dir,v))
            text = ' '.join([w for w in  text_file["TEXT_y"].values[0].split(" ") if w not in stopwords.words('english')])
            text = self.data_processing(text)
            # print(text)
            text_list.append(text)
            y = text_file[self.label_list].values
            label_list.append(y)

            event_file = pd.read_csv(os.path.join(event_patient_id_dir,v))[["procedure_event","input_event_mv","input_event_cv"]].values
            event_codes = []

            for i in range((len(event_file))):
                e = event_file[i]
                for j in e: 
                    if not pd.isnull(j):
                        j = j.lower()
                        words = []
                        for s in j:
                            if s.isalpha():
                                words.append(s)

                            
                        j = "".join(words)
                        # print(j)

                        # j = re.sub(r'[^a-zA-Z\s]', '', j)
                        if j in event_codes: continue
                        event_codes.append(j)
            if not event_codes:
                event_codes.append('Nan')
            event_list.append(event_codes)
        return text_list,label_list,event_list


    def __len__(self):
        return len(self.text_list)


def collate_fn(data):
    text_list = data[0][0]
    label_list = data[0][1]
    event_codes =  data[0][2]
    return text_list,label_list,event_codes

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dataset = PatientDataset('xx',flag="train")
    batch_size = 1
    # model = cw_lstm_model(output=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    label_list = []
    for i,(text_list,label,event_codes_list) in enumerate(tqdm(trainloader)):
        # break
        for d in range(len(text_list)):
            text = text_list[d]

