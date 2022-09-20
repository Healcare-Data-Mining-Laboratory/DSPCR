
import re

from torch.nn import functional as F
import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel
from nltk.tokenize import word_tokenize
from nltk.text import Text
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


data_dir= "dataset/label_desciption.csv"
data_df = pd.read_csv(data_dir)
# label_name = ["Acute and unspecified renal failure",
#         "Acute cerebrovascular disease",
#         "Acute myocardial infarction",
#         "Complications of surgical procedures or medical care",
#         "Fluid and electrolyte disorders",
#         "Gastrointestinal hemorrhage",
#         "Other lower respiratory disease",
#         "Other upper respiratory disease",
#         "Pleurisy; pneumothorax; pulmonary collapse",
#         "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
#         "Respiratory failure; insufficiency; arrest (adult)",
#         "Septicemia (except in labor)",
#         "Shock",
#         "Chronic kidney disease",
#         "Chronic obstructive pulmonary disease and bronchiectasis",
#         "Coronary atherosclerosis and other heart disease",
#         "Diabetes mellitus without complication",
#         "Disorders of lipid metabolism",
#         "Essential hypertension",
#         "Hypertension with complications and secondary hypertension",
#         "Cardiac dysrhythmias",
#         "Conduction disorders",
#         "Congestive heart failure; nonhypertensive",
#         "Diabetes mellitus with complications",
#         "Other liver diseases",
#         ]
label_name = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Complications of surgical procedures or medical care",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Hypertension with complications and secondary hypertension",
    "Cardiac dysrhythmias",
    "Conduction disorders",
    "Congestive heart failure;nonhypertensive",
    "Diabetes mellitus with complications",
    "Other liver diseases",
        ]
def label_descript():
    label_df = data_df.iloc[-25:]
    description_list = label_df["Description"].values.tolist()
    stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())

    description_list = [ word_tokenize(t) for t in description_list]
    new_description_list = []
    for s in description_list:
        temp = []
        for t in s:
            if t.isalpha():
                if t not in stopword:
                    temp.append(t.lower())
        new_description_list.append(" ".join(temp))
    return new_description_list