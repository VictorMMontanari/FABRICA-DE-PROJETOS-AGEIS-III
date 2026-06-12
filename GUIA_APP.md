# 🚀 Guia de Uso: App Streamlit de Predição de Reincidência

## Como Executar o App

### 1. **Certifique-se que o modelo foi treinado**

Primeiro, execute o notebook `projeto9_v2.ipynb` até o final para gerar o arquivo:
```
melhor_modelo_projeto9_v2.joblib
```

### 2. **Instale as dependências (se necessário)**

```bash
pip install streamlit
```

### 3. **Inicie o app**

Na pasta do projeto, execute:

```bash
streamlit run app.py
```

O app abrirá no navegador em `http://localhost:8501`

---

## 📋 Dados Necessários para Fazer uma Predição

A aplicação pede **6 informações principais** sobre uma Ordem de Serviço:

### 1️⃣ **ID da OS** (Texto)
- **Campo**: "🔢 ID da OS"
- **O que é**: Identificador único da Ordem de Serviço
- **Exemplo**: `OS-2024-001`, `OS-12345`, etc.
- **Fonte**: Banco de dados do sistema de gestão de OS
- **Nota**: Usado internamente para "encodar" a OS (representá-la como número)

### 2️⃣ **ID do Consumidor** (Número 1-80000)
- **Campo**: "👤 ID do Consumidor (fake/anonimizado)"
- **O que é**: Número aleatório representando o cliente anonimizado
- **Range**: Valores entre 1 e 80000
- **Exemplo**: 12345, 55678, etc.
- **Fonte**: Campo `consumidor_id_anonimo` do banco de dados original
- **Nota**: No notebook, este é um número gerado aleatoriamente. Em produção, seria o hash do cliente.

### 3️⃣ **Data de Abertura** (Data)
- **Campo**: "📅 Data de Abertura"
- **O que é**: Data em que a OS foi aberta/criada
- **Exemplo**: 01/06/2024
- **Fonte**: Campo `data_abertura` do banco de dados
- **Impacto**: Usada para calcular:
  - Mês de abertura (sazonalidade)
  - Ano de abertura
  - Features circulares (sin/cos do mês)

### 4️⃣ **Data de Fechamento** (Data)
- **Campo**: "📅 Data de Fechamento"
- **O que é**: Data em que a OS foi finalizada
- **Exemplo**: 15/06/2024
- **Fonte**: Campo `data_fechamento` do banco de dados
- **Impacto**: Usada para calcular:
  - Mês de fechamento
  - Ano de fechamento
  - Duração da OS (indiretamente)
- **Validação**: Deve ser **igual ou posterior** à data de abertura

### 5️⃣ **A OS foi Concluída?** (Sim/Não)
- **Campo**: "✅ A OS foi concluída?"
- **O que é**: Indicador de sucesso da OS (atendimento realizado com sucesso)
- **Opções**: 
  - ✅ Sim (1) → Atendimento foi bem-sucedido
  - ❌ Não (0) → Atendimento não foi bem-sucedido ou incompleto
- **Fonte**: Campo `concluida` do banco de dados (0 ou 1)
- **Impacto**: Feature importante para predição
  - OS não concluída tem maior probabilidade de reincidência
  - Captura situações onde o cliente retorna por insatisfação

### 6️⃣ **Cliente tem histórico?** (Sim/Não)
- **Campo**: "Cliente tem histórico de OS anteriores?"
- **O que é**: Se o cliente já abriu OS antes neste sistema
- **Opções**: 
  - ✅ Sim, tem histórico → Ativa campo de "Dias desde última OS"
  - ❌ Não, é novo → Usa valor padrão -1
- **Fonte**: Verificação no histórico do banco de dados
- **Impacto**: Cria flag `sem_historico`:
  - 0 = tem histórico
  - 1 = sem histórico

### 7️⃣ **Dias desde última OS** (Número 0-730, opcional)
- **Campo**: "⏱️ Dias desde última OS"
- **O que é**: Quantos dias passaram entre a última OS e esta
- **Range**: Valores entre 0 e 730 dias (~2 anos)
- **Exemplo**: 
  - 30 dias → cliente retornou após ~1 mês
  - 90 dias → cliente retornou após ~3 meses
  - 365 dias → cliente retornou após ~1 ano
- **Fonte**: Calculado com base no histórico de OS do cliente
- **Nota**: Aparece **apenas se "Tem histórico" = Sim**
- **Se "Não tem histórico"**: Automaticamente usa -1

---

## 📊 Fluxo Completo de Dados

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ENTRADA: Dados Brutos do Usuário                        │
│   ├─ ID da OS                                              │
│   ├─ ID do Consumidor                                      │
│   ├─ Data de Abertura                                      │
│   ├─ Data de Fechamento                                    │
│   ├─ A OS foi Concluída?                                   │
│   ├─ Tem Histórico?                                        │
│   └─ Dias desde Última OS                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. PROCESSAMENTO: Engenharia de Features                    │
│   Cria novas variáveis:                                    │
│   ├─ mes_abertura (1-12)                                  │
│   ├─ ano_abertura (2024, 2025, etc.)                      │
│   ├─ mes_fechamento (1-12)                                │
│   ├─ ano_fechamento (2024, 2025, etc.)                    │
│   ├─ mes_sin = sin(2π × mes / 12)  [sazonalidade circular]│
│   ├─ mes_cos = cos(2π × mes / 12)  [sazonalidade circular]│
│   ├─ dias_desde_ultima_os (valor numérico)                │
│   ├─ sem_historico (0 ou 1)                               │
│   └─ os_id_anonimo (hash simplificado)                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. NORMALIZAÇÃO: StandardScaler                             │
│   Transforma cada feature para:                            │
│   ├─ Média = 0                                             │
│   ├─ Desvio padrão = 1                                    │
│   Colunas normalizadas:                                    │
│   ├─ consumidor_id_fake                                   │
│   ├─ mes_sin                                              │
│   ├─ mes_cos                                              │
│   ├─ dias_desde_ultima_os                                 │
│   ├─ mes_abertura                                         │
│   ├─ ano_abertura                                         │
│   ├─ mes_fechamento                                       │
│   └─ ano_fechamento                                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODELO: MultiOutputClassifier                           │
│   Entrada: 10 features normalizadas                        │
│   Saída: 3 predições                                       │
│   ├─ 30_dias (0 ou 1)                                     │
│   ├─ familia_descricao (classe 0-N)                       │
│   └─ defeito_constatado_descricao (classe 0-N)            │
│   Com: Probabilidades para cada predição                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RESULTADO: Predições e Interpretação                    │
│   ├─ Risco de Reincidência (30 dias)                      │
│   │  └─ Probabilidade e Nível de Confiança               │
│   ├─ Família de Produto com Maior Risco (ID)             │
│   │  └─ Probabilidade de ser essa família                 │
│   └─ Tipo de Defeito com Maior Risco (ID)                │
│      └─ Probabilidade de ser esse defeito                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Interpretando os Resultados

### **Target 1: Reincidência em 30 Dias**

**Predição = 1 (Reincidência = SIM)**
- ⚠️ **Alto risco**: O cliente provavelmente abrirá outra OS para este mesmo produto nos próximos 30 dias
- **Ação**: Revisar diagnóstico, confirmar com cliente, agendar follow-up preventivo
- **Confiança**: % de certeza do modelo nesta predição

**Predição = 0 (Reincidência = NÃO)**
- ✅ **Baixo risco**: Improvável que o cliente retorne nos próximos 30 dias
- **Ação**: Monitoramento padrão
- **Confiança**: % de certeza do modelo nesta predição

---

### **Target 2: Família do Produto**

- **Classe Predita (ID)**: Número que representa qual família de produto tem maior risco
- **Confiança**: Probabilidade de ser aquela família (0-100%)
- **Como usar**: Comparar o ID com a tabela de famílias original do banco de dados

**Exemplo:**
- Se output = "Classe 5" com 75% de confiança
- → Significa: "Há 75% de chance de a reincidência ser da família com ID=5"

---

### **Target 3: Tipo de Defeito**

- **Classe Predita (ID)**: Número que representa qual tipo de defeito tem maior risco
- **Confiança**: Probabilidade de ser aquele defeito (0-100%)
- **Como usar**: Comparar o ID com a tabela de defeitos original do banco de dados

**Exemplo:**
- Se output = "Classe 12" com 68% de confiança
- → Significa: "Há 68% de chance de o defeito de reincidência ser do tipo com ID=12"

---

## 💡 Exemplos Práticos

### **Exemplo 1: Cliente Novo com Risco Baixo**

**Entrada:**
- ID da OS: `OS-2024-5432`
- ID do Consumidor: 22100
- Data de Abertura: 01/06/2024
- Data de Fechamento: 10/06/2024
- **A OS foi Concluída?: Sim (1)**
- Tem Histórico: **Não** (novo cliente)
- Dias desde Última OS: -1 (automático)

**Resultado Esperado:**
- Reincidência 30 dias: **Não (85% confiança)**
- Família do Produto: Classe 2 (72% confiança)
- Tipo de Defeito: Classe 7 (65% confiança)

**Ação:** Monitoramento padrão, sem necessidade de follow-up urgente

---

### **Exemplo 2: Cliente Antigo com Risco Alto**

**Entrada:**
- ID da OS: `OS-2024-5433`
- ID do Consumidor: 45678
- Data de Abertura: 05/06/2024
- Data de Fechamento: 12/06/2024
- **A OS foi Concluída?: Não (0)** ⚠️ (não foi bem-sucedido)
- Tem Histórico: **Sim**
- Dias desde Última OS: 15 (retornou rapidamente)

**Resultado Esperado:**
- Reincidência 30 dias: **Sim (78% confiança)** ⚠️
- Família do Produto: Classe 1 (81% confiança)
- Tipo de Defeito: Classe 3 (76% confiança)

**Ação:**
- ✅ Revisar imediatamente o diagnóstico da OS anterior
- ✅ **A OS não foi concluída** - cliente saiu insatisfeito
- ✅ Entrar em contato com cliente para confirmar satisfação
- ✅ Agendar follow-up em 20 dias para validação preventiva
- ✅ Analisar padrão: cliente tem reincidências frequentes?
- ✅ Verificar se há fila para completar o atendimento

---

## 🔧 Dados Técnicos - Features Utilizadas pelo Modelo

O modelo espera **exatamente 11 features** (na ordem):

| # | Nome da Feature | Tipo | Range Típico | Origem |
|---|-----------------|------|-------------|--------|
| 1 | `os_id_anonimo` | Encoded Integer | 0 - 10000 | Hash do ID da OS |
| 2 | `concluida` | Integer | 0 ou 1 | Entrada do usuário (sucesso do atendimento) |
| 3 | `consumidor_id_fake` | Normalized Float | -1 a 3 | Entrada do usuário |
| 4 | `mes_abertura` | Normalized Float | -1.5 a 1.5 | Mês da data de abertura |
| 5 | `ano_abertura` | Normalized Float | -1 a 3 | Ano da data de abertura |
| 6 | `mes_fechamento` | Normalized Float | -1.5 a 1.5 | Mês da data de fechamento |
| 7 | `ano_fechamento` | Normalized Float | -1 a 3 | Ano da data de fechamento |
| 8 | `mes_sin` | Normalized Float | -1 a 1 | sin(2π × mês / 12) |
| 9 | `mes_cos` | Normalized Float | -1 a 1 | cos(2π × mês / 12) |
| 10 | `dias_desde_ultima_os` | Normalized Float | -2 a 2 | Intervalo entre OS |
| 11 | `sem_historico` | Integer | 0 ou 1 | Flag: tem histórico? |

---

## ⚠️ Limitações e Considerações

1. **Mapping de IDs**
   - Os targets 2 e 3 retornam IDs numéricos (0, 1, 2, etc.)
   - Para traduzir para nomes reais, você precisa do mapping original do `LabelEncoder`
   - Sugestão: Criar tabela de lookup no banco de dados

2. **StandardScaler Simplificado**
   - O app treina um novo StandardScaler a cada predição
   - **Em produção**, deveria usar o mesmo scaler do treinamento
   - Para melhorar: Salve o scaler em `melhor_scaler_projeto9_v2.pkl`

3. **Encoding de OS**
   - Atualmente usa hash simples
   - **Em produção**, deveria usar o LabelEncoder original
   - Para melhorar: Salve o encoder em `melhor_encoder_os.pkl`

4. **Janela de 30 dias**
   - O modelo está treinado para 30 dias
   - Se precisar 60 ou 90 dias, retreine o modelo no notebook

5. **Dados Fora da Distribuição**
   - Se os valores das features estiverem muito fora do intervalo de treino, a predição pode ser menos confiável
   - Verifique os "ranges típicos" da tabela acima

---

## 🚀 Próximas Melhorias

### Curto Prazo (Recomendado):
1. Salvar `StandardScaler` e `LabelEncoder` após treinamento
2. Carregar esses objetos no app (em vez de treinar novos)
3. Criar tabela de mapping para traduzir IDs de produto/defeito para nomes

### Médio Prazo:
1. Integração com banco de dados (carregar dados reais)
2. Histórico de predições (salvar em database)
3. Dashboard de métricas (acurácia por mês, etc.)

### Longo Prazo:
1. API REST para integração com outros sistemas
2. Retrainamento automático periódico
3. Monitoramento de drift de dados

---

## 📞 Suporte

Para dúvidas:
1. Consulte o [README.md](README.md) para detalhes técnicos
2. Revise o notebook `projeto9_v2.ipynb` para entender o processo
3. Verifique a seção "Debug" no app (expanda no final) para ver features brutos
