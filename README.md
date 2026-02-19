# 🚰 Análise Multivariada Aplicada a Indicadores de Saneamento

Projeto acadêmico desenvolvido na disciplina de Análise Multivariada com foco na aplicação de técnicas estatísticas para estudo de indicadores relacionados ao saneamento básico.

Autor: **Arthur Pereira Gon**

---

## 🎯 Objetivo

Investigar a estrutura de dependência entre variáveis associadas ao saneamento básico e identificar padrões estruturais nos dados utilizando técnicas de Análise Multivariada.

O estudo busca:

- Reduzir dimensionalidade
- Identificar fatores latentes
- Interpretar relações entre variáveis
- Avaliar estrutura de correlação

---

## 🗂️ Base de Dados

O conjunto de dados contém indicadores relacionados a saneamento, infraestrutura e condições associadas.

As variáveis analisadas incluem indicadores como:

- Abastecimento de água
- Coleta de esgoto
- Tratamento de esgoto
- Indicadores estruturais relacionados
- Outras variáveis quantitativas associadas ao saneamento

*(Os nomes específicos das variáveis estão no notebook do projeto.)*

---

## 🔎 Metodologia

A análise foi conduzida nas seguintes etapas:

1. Análise descritiva (médias e desvios padrão)
2. Construção da matriz de correlação
3. Verificação da adequação da análise (KMO)
4. Análise de Componentes Principais (PCA)
5. Critério de Kaiser (autovalores > 1)
6. Scree Plot
7. Interpretação das cargas fatoriais
8. Análise dos escores fatoriais

---

## 📊 Técnicas Estatísticas Utilizadas

- Estatística Descritiva
- Matriz de Correlação
- Teste KMO
- Análise de Componentes Principais (PCA)
- Rotação Varimax
- Análise de Escores

---

## 📈 Principais Resultados

- Identificação de fatores principais explicando a maior parte da variabilidade dos dados.
- Redução eficiente da dimensionalidade.
- Identificação de agrupamentos estruturais entre indicadores de saneamento.
- Interpretação de fatores associados a infraestrutura e cobertura de serviços.

---

## 🛠️ Tecnologias Utilizadas

O projeto foi desenvolvido em:

- **Python (Google Colab)**

Principais bibliotecas:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
```

---

## 📦 Estrutura do Projeto

```
├── TRABALHO 2 AM.ipynb
├── dados_saneamento.csv
├── Relatorio_Saneamento.pdf
└── README.md
```

---

## 📌 Conclusão

A aplicação de técnicas multivariadas permitiu:

- Compreender a estrutura interna dos indicadores de saneamento
- Identificar fatores latentes relevantes
- Reduzir complexidade sem perda significativa de informação
- Melhorar a interpretabilidade dos dados

---

## 🚀 Possíveis Extensões

- Aplicação de Cluster Analysis
- Modelos de regressão com escores fatoriais
- Comparação entre regiões
- Análise temporal dos indicadores

---

## 🏫 Contexto Acadêmico

Projeto desenvolvido na disciplina de Análise Multivariada com foco na aplicação prática de métodos estatísticos em dados reais.
