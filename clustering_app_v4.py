
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Titre de l'application
st.title('Clustering des Clients')

# Initialisation d'une liste pour stocker les DataFrames
dataframes = []

# Demandez le nombre de lignes à saisir
num_rows = st.number_input("Combien de lignes souhaitez-vous saisir ?", min_value=1, value=1)

# Boucle pour saisir les données
for i in range(num_rows):
    st.subheader(f"Ligne {i + 1}")
    
    # Saisie des données pour une ligne
    invoice_no = st.text_input(f"Ligne {i + 1} - InvoiceNo :")
    stock_code = st.text_input(f"Ligne {i + 1} - StockCode :")
    description = st.text_input(f"Ligne {i + 1} - Description :")
    quantity = st.number_input(f"Ligne {i + 1} - Quantity :", min_value=0)
    invoice_date = st.date_input(f"Ligne {i + 1} - InvoiceDate :")
    unit_price = st.number_input(f"Ligne {i + 1} - UnitPrice :", min_value=0.0)
    customer_id = st.text_input(f"Ligne {i + 1} - CustomerID :")
    country = st.text_input(f"Ligne {i + 1} - Country :")

    # Ajoutez les données dans une ligne de DataFrame
    data = {
        'InvoiceNo': invoice_no,
        'StockCode': stock_code,
        'Description': description,
        'Quantity': quantity,
        'InvoiceDate': invoice_date,
        'UnitPrice': unit_price,
        'CustomerID': customer_id,
        'Country': country
    }

    # Créez un DataFrame à partir de la ligne de données et ajoutez-le à la liste
    dataframes.append(pd.DataFrame(data, index=[0]))

if dataframes:
    st.subheader('DataFrame créé à partir des données saisies :')
    dataset = pd.concat(dataframes, ignore_index=True)
    st.write(dataset)

    # Créer un bouton pour exécuter le modèle K-Means
    if st.button("Exécuter le modèle K-Means"):
     # Prétraiter les données pour calculer recency, Frequency, et Monetary_value
        #afficher colonne Monetary_value
        dataset['Monetary_value']=dataset['Quantity']*dataset['UnitPrice']
        dataset_m=dataset.groupby('CustomerID')['Monetary_value'].sum()

    	#afficher colonne Frequency
        dataset_fr=dataset.groupby('CustomerID')['InvoiceNo'].nunique()
        dataset_fr = dataset_fr.reset_index()
        dataset_fr.columns = ['CustomerID', 'Frequency']

    	#join les deux datatset
        rfm_final=pd.merge(dataset_m, dataset_fr, on='CustomerID', how='inner')

    	#afficher colonne recency

    	#Convertir la valeur de date maximale en datetime
        last_max_date = pd.to_datetime('2011-12-09 12:49:00')
    	# Trouver la date maximale parmi les dates d'InvoiceDate
        dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
        max_invoice_date =max(dataset['InvoiceDate'])
    	# Sélectionner la date maximale entre les deux
        max_date = max(last_max_date, max_invoice_date)

        rfm_final['recency'] = (max_date - dataset.groupby('CustomerID')['InvoiceDate'].transform('max')).dt.days
        
	#Scaling
        scaler = MinMaxScaler()
        cols_to_scale = ['Monetary_value', 'Frequency', 'recency']
        rfm_df_scaled = scaler.fit_transform(rfm_final[cols_to_scale])

        # Convertir rfm_df_scaled en dataframe
        rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
        rfm_df_scaled.columns = ['Monetary_value', 'Frequency', 'recency']

        # Sélectionner les caractéristiques pertinentes pour le clustering
        X = rfm_df_scaled[['recency', 'Frequency', 'Monetary_value']]

        # Charger le modèle K-Means préalablement entraîné
        k_means = pickle.load(open('model_kmeans.pkl', 'rb'))

        # Attribuer des clusters aux acheteurs
        clusters = k_means.predict(X)

        # Ajouter les étiquettes des clusters attribués en tant que colonne au DataFrame de données
        rfm_final['Cluster'] = clusters

        # Afficher le cluster pour chaque CustomerID
        st.subheader('Clusters attribués aux clients :')
        for customer_id, cluster in zip(rfm_final['CustomerID'], rfm_final['Cluster']):
            st.write(f"Le cluster du Customer ID = '{customer_id}' est Cluster {cluster}")
