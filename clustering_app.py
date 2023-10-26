

import pickle
# Charger le modèle K-Means préalablement entraîné
pickle.dump(k_means_3,open('model_kmeans.pkl','wb'))
k_means=pickle.load(open('model_kmeans.pkl','rb'))

import streamlit as st
import pandas as pd

# Titre de l'application
st.title('Clustering des Clients')

# Charger le fichier de données
data_file = st.file_uploader("Téléchargez le fichier Excel", type=["xls", "xlsx"])

if data_file is not None:
    # Lire le fichier de données
    dataset = pd.read_excel(data_file)

    # Prétraiter les données comme vous l'avez fait précédemment pour calculer recency, Frequency, et Monetary_value
    #afficher colonne Monetary_value
    dataset['Monetary_value']=dataset['Quantity']*dataset['UnitPrice']
    dataset_m=dataset.groupby('CustomerID')['Monetary_value'].sum()

    #afficher colonne Frequency
    dataset_fr=dataset.groupby('CustomerID')['InvoiceNo'].nunique()
    dataset_fr = dataset_fr.reset_index()
    dataset_fr.columns = ['CustomerID', 'Frequency']

    # join les deux datatset
    rfm_final=pd.merge(dataset_m, dataset_fr, on='CustomerID', how='inner')

    #afficher colonne recency

    # Convertir la valeur de date maximale en datetime
    last_max_date = pd.to_datetime('2011-12-09 12:49:00')
    # Trouver la date maximale parmi les dates d'InvoiceDate
    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
    max_invoice_date =max(dataset['InvoiceDate'])
    # Sélectionner la date maximale entre les deux
    max_date = max(last_max_date, max_invoice_date)

    rfm_final['recency'] = (max_date - dataset.groupby('CustomerID')['InvoiceDate'].transform('max')).dt.days

    #Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # Sélectionnez les colonnes à mettre à l'échelle
    cols_to_scale = ['Monetary_value', 'Frequency', 'recency']
    # Appliquez la mise à l'échelle aux colonnes sélectionnées
    rfm_df_scaled= scaler.fit_transform(rfm_final[cols_to_scale])

    # Sélectionner les caractéristiques pertinentes pour le clustering
    X = rfm_df_scaled[['recency', 'Frequency', 'Monetary_value']]

    # Attribuer des clusters aux acheteurs
    clusters = k_means.predict(X)

    # Ajouter les étiquettes des clusters attribués en tant que colonne au DataFrame de données
    dataset['Cluster'] = clusters

    # Afficher le cluster pour chaque CustomerID
    st.subheader('Clusters attribués aux clients :')
    for customer_id, cluster in zip(dataset['CustomerID'], dataset['Cluster']):
        st.write(f"Le cluster du Customer ID = '{customer_id}' est Cluster {cluster}")