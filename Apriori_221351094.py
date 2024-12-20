import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('Mall_Customers.csv')

df.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'Score'
}, inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)

st.header("Isi Dataset")
st.write(X)

clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

st.pyplot(fig)

st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2, 10, 3, 1)

def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust, random_state=42).fit(X)
    X['Labels'] = kmean.labels_

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='Income', y='Score', hue='Labels', size='Labels', data=X, 
                    palette=sns.color_palette('hls', n_clust), ax=ax)

    for label in X['Labels'].unique():
        ax.annotate(label,
                    (X[X['Labels'] == label]['Income'].mean(),
                     X[X['Labels'] == label]['Score'].mean()),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=20, weight='bold', color='black')

    st.header('Cluster Plot')
    st.pyplot(fig)
    st.write(X)

k_means(clust)
