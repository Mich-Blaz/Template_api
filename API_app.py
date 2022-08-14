# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020
@author: MichBlaz
"""

# On importe les libraries necessaire poyue le bon fonctionnement du code
import uvicorn
from fastapi import FastAPI
from Client_datas import Client_data
import numpy as np
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
# Creation de l app et import des choses necessaores
app = FastAPI()

# loader du classifier et de l'explainer
with open("final__pipeline_clf.pkl","rb") as file:
    clf = joblib.load(file)





filename_expl = 'explainer.sav'

with open(filename_expl, 'rb') as file:
    load_explainer = pickle.load()

df_nn=pd.read_csv('data_to_api.csv')

model=clf['classifier']
preproc=clf['preprocessor']


#on cree le Neighbors pour après retourner les clients les plus proches


neigh = NearestNeighbors(n_neighbors=30)
neigh.fit(preproc.transform(df_nn.drop('TARGET',axis=1))

#on peut commencer à faire la fonction qui appelle le Client_Data

@app.post('/get_all_data')

def predict_credit(data:Client_data):


    df_client=pd.DataFrame(data.dict(),index=[0])

    xclient=preproc.transform(df_client)
    idx = neigh.kneighbors(X=xclient,
                           n_neighbors=20,
                           return_distance=False).ravel()
    prediction_proba = model.predict_proba(xclient)
    pred = model.predict(xpred)


    shap_val = dict(pd.Series(load_explainer.shap_values(xclient)[0][0],index=df_client.columns)


    return {
        'pred_class': int(pred[0]),
        'pred_proba':prediction_proba.max(),
        'shap_val':shap_val,
        'Nearest_client': df_nn.iloc[idx].to_dict()
    }



if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload

















