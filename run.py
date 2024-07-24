import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head(10))

print(iris.DESCR)

print("Ознаки (features):", iris.feature_names)

print("Мітки (target):", iris.target_names)

print("Розмірність даних:", iris.data.shape)

print(df.describe())

import seaborn as sns

sns.pairplot(df, hue='target', palette="tab10", diag_kind='hist')

features = df.iloc[:, :-1]
features.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled, columns=features.columns)
df_scaled.head()

from sklearn.cluster import SpectralClustering

clustering = SpectralClustering(n_clusters=3, assign_labels='cluster_qr', random_state=24).fit(df)
sk_clusters_labels = clustering.labels_
df_scaled['cluster'] = sk_clusters_labels

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df['target'], df_scaled['cluster'])

print(cm)

import seaborn as sns

sns.pairplot(df_scaled, hue='cluster', palette="tab10", diag_kind='hist')
