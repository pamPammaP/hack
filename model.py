from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score 
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ident_clust = False

def choising_n_clusters_sil(data, model):
    ssr=[]
    for i in range(2,10):
        models=model(n_clusters=i)
        models.fit_predict(data)
        ssr.append(silhouette_score(data, models.labels_, metric='euclidean'))
    max_score = max(ssr)
    n_clusters = ssr.index(max_score) + 2
    return n_clusters

def choising_n_clusters_inertia(data, model):
  ssr=[]
  for i in range(2,10):
    kmeans=model(n_clusters=i)
    kmeans.fit_predict(data)
    ssr.append(kmeans.inertia_)
  try:
    for i in range(2, len(ssr)):
      if ssr[i-2]/ssr[i-1] > 1.25: # оставляем оптимальное количество количестов кластеров 
        n_clusters = i
  except:
    n_clusters = 1
  return n_clusters

def choising_n_clusters_elbow(data, model): #Elbow Method
  Elbow_M = KElbowVisualizer(model(), k=10)
  Elbow_M.fit(data)
  return Elbow_M.elbow_value_

def choising_n_clusters(df, model):
  global ident_clust
  if model == KMeans:
    n_clusters_int = choising_n_clusters_inertia(df, model)
  n_clusters_elb = choising_n_clusters_elbow(df, model)
  n_clusters_sil = choising_n_clusters_sil(df, model)
  if n_clusters_elb == n_clusters_sil:
    ident_clust = True
    return n_clusters_sil
  return n_clusters_sil

def kmeans_clust(df):
  n = choising_n_clusters(df, KMeans)
  kmeans = KMeans(n_clusters=n)
  labels=kmeans.fit_predict(df)
  score = silhouette_score(df, labels, metric='euclidean')
  return labels, score

def dbscan_clust(df):
  dbscan = DBSCAN()
  labels=dbscan.fit_predict(df)
  score = silhouette_score(df, labels, metric='euclidean')
  return labels, score

def aggl_clust(df):
  n = choising_n_clusters(df, AgglomerativeClustering)
  agg = AgglomerativeClustering(n_clusters=n)
  labels=agg.fit_predict(df)
  score = silhouette_score(df, labels, metric='euclidean')
  return labels, score

def tsne_visiual_paint(df, labels):
  tsne = TSNE(n_components=2,verbose=1,random_state=123)
  z=tsne.fit_transform(df)
  data=pd.DataFrame()
  data['comp-1']=z[:,0]
  data['comp-2']=z[:,1]
  data['labels']=labels
  return data

def choice_model(df):
   return True
   
def main_clustering(df, flag = 0):
  if flag == 0:
    labels_kmeans, score_kmeans = kmeans_clust(df)
    db_score_kmeans = davies_bouldin_score(df , labels_kmeans)
  labels_dbscan, score_dbscan = dbscan_clust(df)
  db_score_dbscan =davies_bouldin_score(df , labels_dbscan)
  labels_aggl, score_aggl = aggl_clust(df)
  db_score_aggl =davies_bouldin_score(df , labels_aggl)
  list_mod_labels = ((db_score_kmeans, labels_kmeans), 
                     (db_score_dbscan, labels_dbscan),
                     (db_score_aggl, labels_aggl))
  labels = labels_kmeans
  score = db_score_kmeans
  for i, j in list_mod_labels:
    if score > i:
      labels = j
      score = i
  #data_tsne = tsne_visiual_paint(df, labels) # возможно понадобится для визуализации
  print(f" Оценка 'силуета' модели kmeans: {score_kmeans} davies - bouldin: {db_score_kmeans}")
  print(f" Оценка 'силуета' модели dbscan: {score_dbscan} davies - bouldin: {db_score_dbscan}")
  print(f" Оценка 'силуета' модели aggl: {score_aggl} davies - bouldin: {db_score_aggl}")
  print(f" Оценка итоговая {score}")
  return labels

def group_clusters(df):
  list_df = []
  for i in df['Cluster'].unique():
    list_df.append(df[df['Cluster'] == i].drop('Cluster', axis=1))
  return list_df


