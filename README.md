# Proposta 9: Predição de Reincidência de Atendimento

## 📋 Visão Geral

Este projeto implementa um **modelo de classificação multi-output** que prevê a probabilidade de reincidência (reabertura) de Ordens de Serviço (OS) após fechamento, além de identificar a família de produtos e tipo de defeito mais prováveis de reincidir.

O modelo é treinado com dados históricos de 12–24 meses de OS fechadas e utiliza técnicas de aprendizado de máquina para ajudar na tomada de decisão operacional.

---

## 🎯 Objetivo

Desenvolver um modelo preditivo que identifique antecipadamente:

1. **Target Primário**: Se uma OS terá reincidência em até **30 dias** (pode ser alterado para 60 ou 90)
2. **Target Secundário 1**: Qual **família do produto** terá maior probabilidade de reincidir
3. **Target Secundário 2**: Qual **defeito constatado** terá maior probabilidade de reincidir

---

## 🔍 Problema Real

Após o fechamento de uma OS, uma fração significativa de clientes retorna com novo atendimento para o **mesmo produto** em curto prazo. Isso indica:

- Possível falha no diagnóstico inicial
- Solução inadequada ou incompleta
- Necessidade de revisão de procedimentos
- Impacto financeiro (custos adicionais, insatisfação do cliente)

**Benefício**: Identificar OS de alto risco permite:
- Revisar o atendimento antes do fechamento
- Priorizar acompanhamento pós-atendimento
- Melhorar processos de diagnóstico e solução
- Reduzir custos de reintervenção

---

## 📊 Estrutura de Dados

### Arquivos de Entrada

O projeto utiliza 4 arquivos CSV principais:

| Arquivo | Descrição |
|---------|-----------|
| `export_os_base.csv` | Dados base das Ordens de Serviço |
| `export_produtos.csv` | Mapeamento de produtos e famílias |
| `export_os_defeito_solucao.csv` | Relacionamento entre OS, defeitos e soluções |
| `export_defeitos_constatados.csv` | Descrição dos defeitos encontrados |

### Merge de Dados

Os arquivos são combinados na seguinte sequência:

```
export_os_base.csv
    ↓ merge (produto_id)
export_produtos.csv → família_descricao
    ↓ merge (os_id_anonimo)
export_os_defeito_solucao.csv → defeito_constatado_id
    ↓ merge (defeito_constatado_id)
export_defeitos_constatados.csv → defeito_constatado_descricao
    ↓
Dataset Final
```

---

## 🔧 Processo Implementado

### 1. **Importação e Configuração** 
- Carregamento de bibliotecas principais (pandas, scikit-learn, matplotlib, seaborn)
- Configuração de estilo de gráficos
- Definição de seed para reprodutibilidade

**Bibliotecas principais:**
- `pandas` - Manipulação de dados
- `scikit-learn` - Modelagem e métricas
- `matplotlib` & `seaborn` - Visualizações
- `joblib` - Serialização de modelos

### 2. **Análise Exploratória (EDA)**

#### 2.1 Informações Gerais
- Dimensões do dataset (linhas × colunas)
- Tipo de dados de cada coluna
- Duplicatas

#### 2.2 Valores Faltantes
- Identificação de colunas com dados incompletos
- Visualização em gráfico de barras horizontal
- Decisão de tratamento (drop, fillna, etc.)

#### 2.3 Estatísticas Descritivas
- **Colunas numéricas**: min, max, mean, std, quantis
- **Colunas categóricas**: número de valores únicos, valor mais frequente

#### 2.4 Análise de Outliers
- Boxplots por coluna numérica
- Identificação de valores extremos

#### 2.5 Distribuição do Target (30 dias)
- Contagem de casos (Reinc. vs Não Reinc.)
- Proporção em gráficos de barras e pizza
- **Avaliação de desbalanceamento**: Razão 0:1
- Aplicação de `class_weight='balanced'` nos modelos

#### 2.6 Taxa de Reincidência por Categoria
- Reincidência por **família do produto** (top 15)
- Reincidência por **tipo de defeito** (top 15)
- Visualização em gráficos de barras ordenados

#### 2.7 Sazonalidade
- Taxa de reincidência mensal
- Identificação de padrões temporais

#### 2.8 Matriz de Correlação
- Correlação entre todas as features numéricas
- Heatmap com máscara triangular
- Correlação específica com o target primário (30_dias)

### 3. **Pré-Processamento e Engenharia de Features**

#### 3.1 Criação de Targets de Reincidência
- Ordenação por consumidor e data de abertura
- Cálculo de diferenças temporais entre OS sucessivas
- **Targets criados**:
  - `30_dias` (target primário)
  - `60_dias` (alternativo)
  - `90_dias` (alternativo)

#### 3.2 Features Temporais
- Extração de **mês e ano** de abertura e fechamento
- **Transformação harmônica**: sin(mês) e cos(mês) para capturar sazonalidade circular
- **Dias desde última OS**: para cada consumidor, dias entre OS consecutivas
- **Flag "sem histórico"**: indicador se consumidor é novo

#### 3.3 Filtro Temporal
- Remoção de OS fechadas há menos de 30 dias
- Evita viés temporal (impossibilidade de detectar reincidência)

#### 3.4 Limpeza de Colunas
- **Drop** de colunas irrelevantes:
  - IDs internos (`fabrica_id`, `consumidor_id_anonimo`)
  - Dados temporais duplicados (após extração de features)
  - Targets alternativos (60_dias, 90_dias, se usando apenas 30_dias)
- **Tratamento de valores faltantes**: Remoção de linhas com valores críticos

#### 3.5 Encoding
- **Label Encoding**: Colunas categóricas → números inteiros
  - `os_id_anonimo`
  - `familia_descricao`
  - `defeito_constatado_descricao`
- **Armazenamento de encoders**: Para possível decode futuro

#### 3.6 Normalização
- **StandardScaler** em features numéricas:
  - Centraliza em torno de 0
  - Padroniza variância para 1
  - Crucial para modelos baseados em distância

### 4. **Divisão dos Dados**

**Estratégia**: 40% Treino | 30% Validação | 30% Teste

```
Dataset Original
    ↓ test_size=0.60, stratify=30_dias
├── Treino (40%)
└── Temp (60%)
    ↓ test_size=0.50 (em Temp), stratify=30_dias
    ├── Validação (30%)
    └── Teste (30%)
```

**Características da divisão:**
- **Estratificação** no target primário (preserva proporção de reincidência)
- Cada split possui:
  - Número de amostras
  - Contagem de casos de reincidência
  - Percentual de reincidência

---

## 🤖 Modelos Utilizados

### Multi-Output Classification

O problema é modelado como **classificação multi-output**, onde um único modelo prevê **3 targets simultaneamente**:

```python
MultiOutputClassifier([
    Modelo_para_Target_1 (30_dias),
    Modelo_para_Target_2 (familia_descricao),
    Modelo_para_Target_3 (defeito_constatado_descricao)
])
```

### Modelos Base

#### 1. **Decision Tree Classifier**
```python
DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
```

- **Vantagens**: Interpretável, rápido, não requer normalização
- **Desvantagem**: Propício a overfitting
- **Configuração**: max_depth=10 controla profundidade

#### 2. **Random Forest Classifier**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
```

- **Vantagens**: Robusto, reduz overfitting via ensemble, importância de features
- **Desvantagem**: Menos interpretável, mais lento
- **Configuração**:
  - `n_estimators=100`: 100 árvores no ensemble
  - `max_depth=15`: Maior profundidade que Decision Tree
  - `n_jobs=-1`: Paralelização (usa todos os cores)

---

## 📈 Avaliação e Métricas

### Métricas Implementadas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Accuracy** | (TP + TN) / Total | % de predições corretas (cuidado: enganoso com desbalanceamento) |
| **Precision** | TP / (TP + FP) | % de casos previstos como reincidentes que realmente reincidiram |
| **Recall** | TP / (TP + FN) | % de casos realmente reincidentes que foram identificados |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Média harmônica; melhor para desbalanceamento |
| **AUC-ROC** | Área sob curva ROC | Robustez; varia de 0.5 (aleatório) a 1.0 (perfeito) |

### Análises Realizadas

#### 4.1 Métricas por Split (Validação e Teste)
- Comparação lado-a-lado de todos os modelos
- Identificação do melhor modelo por métrica

#### 4.2 Gráfico Comparativo
- Barras agrupadas por métrica
- Cores diferentes para cada modelo
- Rótulos com valores exatos

#### 4.3 Matrizes de Confusão
- Uma matriz por modelo × split (Validação / Teste)
- Breakdown de TP, TN, FP, FN
- Percentuais de cada célula

#### 4.4 Curvas ROC / AUC
- Gráficos lado-a-lado (Validação vs Teste)
- Comparação visual do desempenho
- Linha de referência (aleatório = diagonal)

#### 4.5 Classification Report
- Precision, Recall, F1 por classe
- Support (número de amostras)
- Macro/Weighted averages

#### 4.6 Importância de Features (Random Forest)
- Top 20 features mais importantes
- Gráfico de barras horizontal
- Ordenação decrescente de importância

#### 4.7 Métricas Multi-Output
- Tabela com Accuracy e F1-Macro para cada target
- Validação de desempenho em todos os 3 targets

---

## 📊 Resultados e Recomendação

### Tabela Comparativa Final

O notebook compila resultados de ambos os modelos em **Validação** e **Teste**, destacando:

- **Melhor Modelo por Métrica**: Qual modelo obteve melhor score em cada métrica
- **Recomendação**: 
  - Para minimizar **falsos negativos** (não detectar reincidência): priorizar **Recall**
  - Para uso em produção: equilibrio entre Precision e Recall (F1-Score)

### Modelo Salvo

O melhor modelo (**Decision Tree**) é automaticamente serializado em:
```
melhor_modelo_projeto9_v2.joblib
```

---

## 💾 Como Usar

### 1. **Executar o Notebook**

```bash
jupyter notebook projeto9_v2.ipynb
```

Execute as células na sequência:
1. **Seção 1**: Importações
2. **Seção 2**: Carregamento e merge de dados
3. **Seção 3**: EDA (exploração)
4. **Seção 4**: Pré-processamento
5. **Seção 5**: Encoding e normalização
6. **Seção 6**: Divisão treino/validação/teste
7. **Seção 7**: Treinamento dos modelos
8. **Seção 8**: Avaliação completa
9. **Seção 9**: Comparação final e salvamento

### 2. **Carregar Modelo Treinado**

```python
import joblib
from pathlib import Path

# Carregar modelo
modelo = joblib.load('melhor_modelo_projeto9_v2.joblib')

# Fazer predições
y_pred = modelo.predict(X_novo)  # shape: (n_amostras, 3)

# y_pred[:, 0] → 30_dias
# y_pred[:, 1] → familia_descricao
# y_pred[:, 2] → defeito_constatado_descricao
```

### 3. **Customizar Janela de Reincidência**

Para alterar a janela de 30 dias para **60 ou 90**:

1. Na célula 4.3, altere:
   ```python
   TARGET_COLS = ['60_dias', 'familia_descricao', 'defeito_constatado_descricao']
   ```

2. Reexecute as seções subsequentes

---

## 📁 Estrutura de Diretórios

```
FABRICA-DE-PROJETOS-AGEIS-III/
├── README.md                              ← Esta documentação
├── projeto9_v2.ipynb                      ← Notebook principal (todo o código)
├── problema.txt                           ← Especificação original do problema
├── requirements.txt                       ← Dependências Python
├── melhor_modelo_projeto9_v2.joblib       ← Modelo serializado (gerado após execução)
├── app.py                                 ← App auxiliar (não usado neste projeto)
└── dataset/
    ├── export_os_base.csv
    ├── export_produtos.csv
    ├── export_os_defeito_solucao.csv
    ├── export_defeitos_constatados.csv
    └── [outros 8 arquivos CSV]
```

---

## 📦 Dependências

```
streamlit==1.40.2
numpy==2.0.2
pandas==2.2.3
joblib==1.4.2
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
jupyter==1.1.1
```

Para instalar:
```bash
pip install -r requirements.txt
```

---

## 🚀 Próximos Passos

### Melhorias Sugeridas

1. **Otimização de Hiperparâmetros**
   - Grid Search ou Random Search em `max_depth`, `n_estimators`
   - Cross-validation para validação mais robusta

2. **Balanceamento de Classes**
   - SMOTE (Synthetic Minority Over-sampling)
   - Ajuste de threshold de classificação

3. **Engenharia de Features Avançada**
   - Features agregadas (histórico por produto, por posto)
   - Interações entre features
   - Análise de componentes principais (PCA)

4. **Modelos Adicionais**
   - Gradient Boosting (XGBoost, LightGBM)
   - Redes Neurais (TensorFlow/Keras)
   - Ensemble com stacking

5. **Validação Temporal**
   - Treino em período anterior, teste em período mais recente
   - Walk-forward validation para séries temporais

6. **Deploys**
   - API REST (Flask, FastAPI)
   - Integração com sistema de gestão de OS
   - Monitoramento de drift de dados

---

## 📝 Notas Importantes

- **Desbalanceamento**: O dataset pode ter proporção desigual de Reincidência vs Não-Reincidência. O parâmetro `class_weight='balanced'` ajusta automaticamente os pesos.

- **Reprodutibilidade**: `random_state=42` garante resultados iguais em múltiplas execuções.

- **Normalização**: StandardScaler é aplicado em features numéricas **antes** do split treino/teste para evitar data leakage.

- **Identificador de Consumidor**: O modelo usa `consumidor_id_fake` gerado aleatoriamente nesta versão. Na produção, usar `consumidor_id_anonimo` real do dataset.

---

## 📧 Contato / Dúvidas

Para dúvidas sobre a implementação, referir-se ao notebook `projeto9_v2.ipynb` onde cada célula possui comentários detalhados.

---

**Última atualização:** Junho 2026  
**Versão:** 2.0  
**Status:** Produção
