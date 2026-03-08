# ============================================================
# Análise Multidimensional da Qualidade do Saneamento em
# São Paulo: Uma Abordagem com Aprendizado de Máquina (2013-2024)
# Autor: Arthur Pereira Gon — UFSCar
# ============================================================
#
# DADOS: Disponíveis publicamente no Base dos Dados (SNIS)
# Link: https://basedosdados.org/dataset/2a543ad8-3cdb-4047-9498-efe7fb8ed697?table=df7cf198-4889-4baf-bb77-4e0e28eb90ca
#
# INSTRUÇÕES PARA BAIXAR OS DADOS:
# 1. Acesse o link acima
# 2. Vá até "Acesso aos dados"
# 3. Clique na aba "Download"
# 4. Baixe o arquivo na linha "id_municipio"
# 5. Baixe a Tabela no botão "Download da Tabela"
# 6. Faça upload dos dois arquivos antes de rodar este script
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ── 1. CARREGANDO E PRÉ-PROCESSANDO OS DADOS ────────────────
# Após fazer upload dos arquivos, ajuste os nomes abaixo
df = pd.read_csv("saneamento_sp.csv")  # ajuste o nome do arquivo baixado

# Filtrando apenas municípios de São Paulo
df_sp = df[df["sigla_uf"] == "SP"].copy()

# Agregando por período (2013-2022) — média por município
df_agg = df_sp.groupby("id_municipio").mean(numeric_only=True).reset_index()

# Selecionando features de saneamento
features = [
    "ind_cobertura_agua",
    "ind_cobertura_esgoto",
    "ind_tratamento_esgoto",
    "rede_esgoto",
    "faturamento_esgoto",
    "investimento_total",
]

df_features_raw = df_agg[features].dropna()

# Normalizando os dados (StandardScaler)
scaler = StandardScaler()
df_features = pd.DataFrame(
    scaler.fit_transform(df_features_raw),
    columns=features,
    index=df_features_raw.index
)

print(f"Municípios analisados: {len(df_features)}")
print(df_features.describe())

# ── 2. ENCONTRANDO O K IDEAL — ÍNDICE DE SILHUETA ───────────
print("\nCalculando o Índice de Silhueta para cada k...")

silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_features)
    score = silhouette_score(df_features, labels)
    silhouette_scores.append(score)
    print(f"Para k={k}, a pontuação da Silhueta é: {score:.4f}")

best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\nO melhor número de clusters (k) encontrado pelo Índice de Silhueta é: {best_k}")

# Gráfico do Índice de Silhueta
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), silhouette_scores, marker='o', color='blue')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Melhor k = {best_k}')
plt.title('Índice de Silhueta para Encontrar o k Ideal')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Índice de Silhueta Médio')
plt.legend()
plt.grid(True)
plt.show()

# ── 3.1 EXECUÇÃO K-MEANS ────────────────────────────────────
print(f"\nRodando K-Means com k={best_k}...")

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(df_features)
df_features_raw["cluster_kmeans"] = labels_kmeans

# Análise dos clusters K-Means
print("\nDistribuição dos clusters (K-Means):")
print(df_features_raw["cluster_kmeans"].value_counts().sort_index())

print("\nPerfil médio de cada cluster (K-Means):")
print(df_features_raw.groupby("cluster_kmeans")[features].mean().round(3))

# ── 3.2 EXECUÇÃO DBSCAN ─────────────────────────────────────
# Encontrando epsilon via k-distance plot
min_samples = 15

print(f"\nCalculando as distâncias para os {min_samples} vizinhos mais próximos...")
vizinhos = NearestNeighbors(n_neighbors=min_samples)
vizinhos_fit = vizinhos.fit(df_features)
distancias, indices = vizinhos_fit.kneighbors(df_features)

distancias = np.sort(distancias[:, -1])

plt.figure(figsize=(10, 6))
plt.plot(distancias)
plt.title(f'Gráfico de Distância para o {min_samples}º Vizinho Mais Próximo (k-distance plot)')
plt.xlabel("Pontos (Cidades), ordenados por distância")
plt.ylabel("Distância Epsilon (eps)")
plt.grid(True)
plt.show()

# Epsilon definido visualmente pelo cotovelo do k-distance plot
epsilon = 4
print(f"\nEpsilon = {epsilon} escolhido e min_samples escolhemos {min_samples}.")

# Rodando DBSCAN
print("\nRodando DBSCAN...")
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels_dbscan = dbscan.fit_predict(df_features)
df_features_raw["cluster_dbscan"] = labels_dbscan

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_outliers = list(labels_dbscan).count(-1)
print(f"Clusters encontrados pelo DBSCAN: {n_clusters_dbscan}")
print(f"Outliers identificados: {n_outliers}")
print("\nDistribuição DBSCAN:")
print(pd.Series(labels_dbscan).value_counts().sort_index())

# ── 3.3 AGRUPAMENTO HIERÁRQUICO ─────────────────────────────
print("\nGerando a matriz de ligação para o dendrograma...")

linked = linkage(df_features, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(
    linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True,
    truncate_mode='lastp',
    p=12
)
plt.title('Dendrograma do Agrupamento Hierárquico')
plt.ylabel('Distância Euclidiana (Ward)')
plt.xlabel('Número de Pontos no Nó (ou Índice do Ponto)')
plt.axhline(y=70, c='red', linestyle='--')
plt.show()

# Rodando Hierárquico com k=3 (definido pelo dendrograma)
k_hierarquico = 3
print(f"\nRodando Agrupamento Hierárquico com k={k_hierarquico}...")

hierarquico = AgglomerativeClustering(n_clusters=k_hierarquico)
labels_hierarquico = hierarquico.fit_predict(df_features)
df_features_raw["cluster_hierarquico"] = labels_hierarquico

print(f"\nDistribuição dos clusters (Hierárquico k={k_hierarquico}):")
print(df_features_raw["cluster_hierarquico"].value_counts().sort_index())

print(f"\nPerfil médio de cada cluster (Hierárquico):")
print(df_features_raw.groupby("cluster_hierarquico")[features].mean().round(3))

# ── 4. COMPARAÇÃO DE MÉTRICAS ────────────────────────────────
print("\n--- Tabela Comparativa de Performance dos Algoritmos ---")

# Métricas K-Means
score_s_kmeans = silhouette_score(df_features, labels_kmeans)
score_ch_kmeans = calinski_harabasz_score(df_features, labels_kmeans)
score_db_kmeans = davies_bouldin_score(df_features, labels_kmeans)

# Métricas Hierárquico
score_s_hierarquico = silhouette_score(df_features, labels_hierarquico)
score_ch_hierarquico = calinski_harabasz_score(df_features, labels_hierarquico)
score_db_hierarquico = davies_bouldin_score(df_features, labels_hierarquico)

# Métricas DBSCAN (apenas core points)
mask_dbscan = labels_dbscan != -1
if sum(mask_dbscan) > 0 and len(set(labels_dbscan[mask_dbscan])) > 1:
    score_s_dbscan = silhouette_score(df_features[mask_dbscan], labels_dbscan[mask_dbscan])
    score_ch_dbscan = calinski_harabasz_score(df_features[mask_dbscan], labels_dbscan[mask_dbscan])
    score_db_dbscan = davies_bouldin_score(df_features[mask_dbscan], labels_dbscan[mask_dbscan])
else:
    score_s_dbscan, score_ch_dbscan, score_db_dbscan = (None, None, None)

resultados = {
    'Algoritmo': [f'K-Means (k={best_k})', f'Hierárquico (k={k_hierarquico})', 'DBSCAN (core points)'],
    'Índice de Silhueta (maior é melhor)': [score_s_kmeans, score_s_hierarquico, score_s_dbscan],
    'Calinski-Harabasz (maior é melhor)': [score_ch_kmeans, score_ch_hierarquico, score_ch_dbscan],
    'Davies-Bouldin (menor é melhor)': [score_db_kmeans, score_db_hierarquico, score_db_dbscan],
}

df_resultados = pd.DataFrame(resultados).set_index('Algoritmo')
print(df_resultados.round(4))
