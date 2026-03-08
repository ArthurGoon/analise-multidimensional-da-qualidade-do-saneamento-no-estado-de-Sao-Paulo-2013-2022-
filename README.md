# Multidimensional Analysis of Sanitation Quality in São Paulo (2013–2024)

Clustering analysis across all 630 São Paulo municipalities to identify sanitation profiles and priority areas for public policy, using unsupervised machine learning on SNIS public data.

**Author:** Arthur Gon &nbsp;|&nbsp; **Stack:** Python &nbsp;|&nbsp; **Techniques:** K-Means, DBSCAN, Hierarchical Clustering

---

## Key Findings

- **549 municipalities** share a homogeneous "São Paulo Standard" profile with balanced sanitation indicators close to the state average
- **62 municipalities** form a critical cluster with severely low sewage coverage (avg. -2.42 std) - the priority group for infrastructure expansion
- **13 municipalities** form a high-efficiency niche with very high per capita water consumption and extremely low billing losses (-3.80 std)
- K-Means (k=8) achieved the best Silhouette Score (0.33), while Hierarchical (k=3) delivered the best Calinski-Harabasz (99.80) for macro-group separation
- DBSCAN identified **39 outliers** - municipalities like Guarulhos, Águas de São Pedro, and Ilha Comprida with structurally distinct profiles

---

## Methodology

Data sourced from SNIS (Sistema Nacional de Informações sobre Saneamento), aggregated across 2013–2022 and normalized per capita for fair comparison across cities.

**Pipeline:**
1. Data loading and preprocessing (SNIS public dataset)
2. Feature engineering - 18 normalized sanitation metrics per municipality
3. Optimal k selection via Silhouette Score analysis (k=2 to 10)
4. K-Means execution with best k
5. DBSCAN with k-distance plot to define epsilon (eps=4, min_samples=15)
6. Hierarchical clustering with dendrogram analysis (Ward linkage, k=3)
7. Algorithm comparison via Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics

---

## Stack

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Data

Public dataset from Base dos Dados (SNIS):
[basedosdados.org/dataset/snis-saneamento](https://basedosdados.org/dataset/2a543ad8-3cdb-4047-9498-efe7fb8ed697?table=df7cf198-4889-4baf-bb77-4e0e28eb90ca)

To reproduce:
1. Access the link above and go to "Acesso aos dados"
2. Click "Download" and download the `id_municipio` file
3. Download the table via "Download da Tabela"
4. Upload both files and adjust the filename in `analise_saneamento.py`

---

## Project Structure

```
├── analise_saneamento.py
├── Saneamento.pdf
└── README.md
```

---

## Possible Extensions

- Temporal analysis of each cluster across 2013–2024 to detect improvement or degradation trends
- Geographic mapping of clusters to identify regional patterns
- Regression model using cluster membership to predict investment needs
- Replication for other Brazilian states
