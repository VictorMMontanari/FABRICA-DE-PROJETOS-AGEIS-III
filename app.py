"""
Aplicação Streamlit para Predição de Reincidência de Atendimento
Proposta 9: Predição de Reincidência de Atendimento

Esta aplicação permite fazer predições de reincidência de Ordens de Serviço (OS)
utilizando o modelo treinado em projeto9_v2.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================================
# CONFIGURAÇÃO PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Predição de Reincidência de OS",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado do arquivo .joblib"""
    modelo_path = Path.cwd() / 'projeto9_v2.joblib'
    
    if not modelo_path.exists():
        st.error(f"❌ Modelo não encontrado em {modelo_path}")
        st.info("Por favor, execute o notebook `projeto9_v2.ipynb` para treinar o modelo.")
        st.stop()
    
    modelo = joblib.load(modelo_path)
    return modelo


def criar_features_a_partir_de_datas(data_abertura, data_fechamento):
    """
    Cria features temporais a partir de duas datas.
    
    Args:
        data_abertura: datetime
        data_fechamento: datetime
    
    Returns:
        dict com features temporais
    """
    features_temp = {}
    
    # Extração de mês e ano
    features_temp['mes_abertura'] = data_abertura.month
    features_temp['ano_abertura'] = data_abertura.year
    features_temp['mes_fechamento'] = data_fechamento.month
    features_temp['ano_fechamento'] = data_fechamento.year
    
    # Transformação harmônica do mês (captura sazonalidade circular)
    features_temp['mes_sin'] = np.sin(2 * np.pi * features_temp['mes_abertura'] / 12)
    features_temp['mes_cos'] = np.cos(2 * np.pi * features_temp['mes_abertura'] / 12)
    
    return features_temp


def processar_entrada_usuario(
    os_id,
    data_abertura,
    data_fechamento,
    dias_desde_ultima_os_valor,
    tem_historico,
    consumidor_id,
    concluida
):
    """
    Processa os dados de entrada do usuário e cria um DataFrame com as features.
    
    IMPORTANTE: A ordem das colunas DEVE corresponder exatamente à ordem esperada pelo modelo:
    ['os_id_anonimo', 'concluida', 'consumidor_id_fake', 'mes_abertura', 'ano_abertura',
     'mes_fechamento', 'ano_fechamento', 'mes_sin', 'mes_cos', 'dias_desde_ultima_os', 'sem_historico']
    
    Args:
        os_id: str - ID da OS (será usado para Label Encoding)
        data_abertura: datetime
        data_fechamento: datetime
        dias_desde_ultima_os_valor: int - dias desde última OS (pode ser -1 se sem histórico)
        tem_historico: bool - se tem histórico de OS
        consumidor_id: int - ID do consumidor (usado para gerar fake ID)
        concluida: int - 1 se concluída, 0 se não (flag de sucesso)
    
    Returns:
        pd.DataFrame com features processadas na ordem correta (ainda não normalizadas)
    """
    
    # Features temporais
    features_temp = criar_features_a_partir_de_datas(data_abertura, data_fechamento)
    
    # Criar DataFrame com features na ORDEM CORRETA esperada pelo modelo
    df_features = pd.DataFrame([{
        'os_id_anonimo': hash(str(os_id)) % 10000,
        'concluida': concluida,
        'consumidor_id_fake': consumidor_id,
        'mes_abertura': features_temp['mes_abertura'],
        'ano_abertura': features_temp['ano_abertura'],
        'mes_fechamento': features_temp['mes_fechamento'],
        'ano_fechamento': features_temp['ano_fechamento'],
        'mes_sin': features_temp['mes_sin'],
        'mes_cos': features_temp['mes_cos'],
        'dias_desde_ultima_os': dias_desde_ultima_os_valor,
        'sem_historico': 0 if tem_historico else 1,
    }])
    
    return df_features


def normalizar_features(df_features):
    """
    Normaliza as features usando StandardScaler.
    
    Nota: Em produção, você deveria carregar o StandardScaler treinado
    do arquivo original de treinamento para evitar dados fora da escala.
    
    Args:
        df_features: pd.DataFrame com features
    
    Returns:
        pd.DataFrame normalizado
    """
    
    # Colunas numéricas que devem ser normalizadas (mesmo padrão do notebook)
    scale_cols = [
        'consumidor_id_fake', 'mes_sin', 'mes_cos',
        'dias_desde_ultima_os', 'mes_abertura', 'ano_abertura',
        'mes_fechamento', 'ano_fechamento',
    ]
    
    scaler = StandardScaler()
    df_features_norm = df_features.copy()
    df_features_norm[scale_cols] = scaler.fit_transform(df_features[scale_cols])
    
    return df_features_norm


# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Título e descrição
st.title("🔮 Predição de Reincidência de Atendimento")
st.markdown("""
    **Previsão de Reabertura de Ordens de Serviço (OS)**
    
    Esta aplicação utiliza um modelo de aprendizado de máquina treinado para prever:
    1. Se uma OS terá reincidência em até **30 dias**
    2. Qual **família de produto** terá maior probabilidade de reincidir
    3. Qual **tipo de defeito** terá maior probabilidade de reincidir
""")

st.divider()

# ============================================================================
# BARRA LATERAL - INFORMAÇÕES
# ============================================================================

with st.sidebar:
    st.header("📊 Sobre o Modelo")
    st.info("""
        **Modelos Utilizados:**
        - Decision Tree Classifier
        - Random Forest Classifier (com MultiOutputClassifier)
        
        **Dados de Treino:**
        - 40% Treino | 30% Validação | 30% Teste
        - Estratificação por classe de reincidência
        
        **Arquivo do Modelo:**
        - `melhor_modelo_projeto9_v2.joblib`
    """)
    
    st.markdown("---")
    
    st.subheader("📖 Documentação")
    st.markdown("""
        Para detalhes completos sobre o projeto, veja:
        - [README.md](file:/README.md) - Documentação completa
        - `projeto9_v2.ipynb` - Código e análises
    """)

# ============================================================================
# SEÇÃO PRINCIPAL - FORMULÁRIO DE ENTRADA
# ============================================================================

st.header("📋 Dados da Ordem de Serviço")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Identificadores")
    os_id = st.text_input(
        "🔢 ID da OS",
        value="OS-2024-001",
        help="Identificador único da Ordem de Serviço"
    )
    
    consumidor_id = st.number_input(
        "👤 ID do Consumidor (fake/anonimizado)",
        min_value=1,
        max_value=80000,
        value=12345,
        help="Número aleatório entre 1 e 80000 representando o cliente anonimizado"
    )

with col2:
    st.subheader("Datas e Status")
    data_abertura = st.date_input(
        "📅 Data de Abertura",
        value=datetime(2024, 6, 1),
        help="Quando a OS foi aberta"
    )
    
    data_fechamento = st.date_input(
        "📅 Data de Fechamento",
        value=datetime(2024, 6, 15),
        help="Quando a OS foi fechada"
    )
    
    concluida = st.radio(
        "✅ A OS foi concluída?",
        options=[1, 0],
        format_func=lambda x: "Sim (1)" if x == 1 else "Não (0)",
        horizontal=True,
        help="Se o atendimento foi realizado com sucesso"
    )

# Validação de datas
if data_fechamento < data_abertura:
    st.error("❌ Data de fechamento não pode ser antes da data de abertura!")
    st.stop()

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Histórico do Cliente")
    
    tem_historico = st.radio(
        "Cliente tem histórico de OS anteriores?",
        options=[True, False],
        format_func=lambda x: "Sim, tem histórico" if x else "Não, é novo",
        help="Se o cliente já abriu OS antes"
    )

with col4:
    st.subheader("Intervalo Temporal")
    
    if tem_historico:
        dias_desde_ultima_os_valor = st.number_input(
            "⏱️ Dias desde última OS",
            min_value=0,
            max_value=730,  # ~2 anos
            value=30,
            help="Quantos dias passaram desde a última OS deste cliente"
        )
    else:
        dias_desde_ultima_os_valor = -1
        st.info("ⓘ Para clientes novos, este campo é automaticamente -1")

st.divider()

# ============================================================================
# BOTÃO DE PREDIÇÃO
# ============================================================================

if st.button("🚀 Fazer Predição", type="primary", use_container_width=True):
    
    # Carregar modelo
    with st.spinner("Carregando modelo..."):
        modelo = carregar_modelo()
    
    # Processar dados
    with st.spinner("Processando dados..."):
        # Converter datas de st.date_input (datetime.date) para datetime.datetime
        data_abertura_dt = pd.to_datetime(data_abertura)
        data_fechamento_dt = pd.to_datetime(data_fechamento)
        
        df_features = processar_entrada_usuario(
            os_id=os_id,
            data_abertura=data_abertura_dt,
            data_fechamento=data_fechamento_dt,
            dias_desde_ultima_os_valor=dias_desde_ultima_os_valor,
            tem_historico=tem_historico,
            consumidor_id=consumidor_id,
            concluida=concluida
        )
        
        df_features_norm = normalizar_features(df_features)
    
    # Fazer predição
    with st.spinner("Fazendo predição..."):
        predicoes = modelo.predict(df_features_norm)
        probabilidades = [
            modelo.predict_proba(df_features_norm)[i][:, 1]
            for i in range(3)
        ]
    
    st.divider()
    
    # ========================================================================
    # EXIBIR RESULTADOS
    # ========================================================================
    
    st.header("✅ Resultados da Predição")
    
    # Resultado 1: Reincidência em 30 dias
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.subheader("🔴 Target 1: Reincidência em 30 Dias")
        
        pred_30dias = predicoes[0][0]
        prob_30dias = probabilidades[0][0]
        
        if pred_30dias == 1:
            st.error(f"**⚠️ ALTO RISCO DE REINCIDÊNCIA**")
            st.metric(
                "Predição",
                "Sim (Reincidirá)",
                f"Confiança: {prob_30dias*100:.1f}%",
                delta=f"+{prob_30dias*100:.1f}%",
                delta_color="inverse"
            )
        else:
            st.success(f"**✓ Baixo Risco de Reincidência**")
            st.metric(
                "Predição",
                "Não (Não Reincidirá)",
                f"Confiança: {(1-prob_30dias)*100:.1f}%"
            )
        
        # Barra de progresso
        st.progress(prob_30dias, text=f"Probabilidade de Reincidência: {prob_30dias*100:.1f}%")
    
    with col_r2:
        st.info("""
            **Interpretação:**
            - **Reincidência = 1**: Cliente provavelmente abrirá nova OS nos próximos 30 dias
            - **Reincidência = 0**: Improvável que cliente abra nova OS nos próximos 30 dias
            
            **Ação recomendada:**
            - Se ALTO RISCO: Revisar solução, confirmar com cliente, agendar follow-up
            - Se BAIXO RISCO: Monitoramento padrão
        """)
    
    st.divider()
    
    # Resultado 2: Família do Produto
    st.subheader("🏭 Target 2: Família de Produto com Maior Risco")
    
    pred_familia = predicoes[0][1]
    prob_familia = probabilidades[1][0]
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.metric(
            "Classe Predita (ID)",
            f"{pred_familia}",
            help="ID do encoder da família de produto"
        )
    
    with col_f2:
        st.metric(
            "Confiança",
            f"{prob_familia*100:.1f}%"
        )
    
    with col_f3:
        st.info("ℹ️ Este ID corresponde a uma das famílias de produtos do dataset.\nConsulte o mapping original para traduzir para nome da família.")
    
    st.divider()
    
    # Resultado 3: Tipo de Defeito
    st.subheader("⚙️ Target 3: Tipo de Defeito com Maior Risco")
    
    pred_defeito = predicoes[0][2]
    prob_defeito = probabilidades[2][0]
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.metric(
            "Classe Predita (ID)",
            f"{pred_defeito}",
            help="ID do encoder do tipo de defeito"
        )
    
    with col_d2:
        st.metric(
            "Confiança",
            f"{prob_defeito*100:.1f}%"
        )
    
    with col_d3:
        st.info("ℹ️ Este ID corresponde a um dos defeitos do dataset.\nConsulte o mapping original para traduzir para descrição do defeito.")
    
    st.divider()
    
    # ========================================================================
    # RESUMO EM TABELA
    # ========================================================================
    
    st.subheader("📊 Resumo das Predições")
    
    resumo_df = pd.DataFrame({
        "Target": [
            "Reincidência 30 dias",
            "Família do Produto",
            "Tipo de Defeito"
        ],
        "Predição": [
            "Sim" if pred_30dias == 1 else "Não",
            f"Classe {int(pred_familia)}",
            f"Classe {int(pred_defeito)}"
        ],
        "Confiança (%)": [
            f"{prob_30dias*100:.2f}%",
            f"{prob_familia*100:.2f}%",
            f"{prob_defeito*100:.2f}%"
        ]
    })
    
    st.dataframe(resumo_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ========================================================================
    # DEBUG - FEATURES UTILIZADAS
    # ========================================================================
    
    with st.expander("🔧 Debug - Features Utilizadas (apenas para desenvolvimento)"):
        st.write("**Features sem normalização:**")
        st.dataframe(df_features, use_container_width=True)
        
        st.write("**Features normalizadas (entrada do modelo):**")
        st.dataframe(df_features_norm, use_container_width=True)
        
        st.write("**Ordem das features:**")
        st.code(str(list(df_features_norm.columns)))

# ============================================================================
# RODAPÉ
# ============================================================================

st.divider()

st.markdown("""
    ---
    
    **📚 Sobre os Dados de Entrada:**
    
    O modelo espera as seguintes informações sobre uma Ordem de Serviço:
    
    | Campo | Tipo | Descrição |
    |-------|------|-----------|
    | ID da OS | String | Identificador único da OS |
    | ID do Consumidor | Número (1-80000) | ID anonimizado do cliente |
    | Data de Abertura | Data | Quando a OS foi aberta |
    | Data de Fechamento | Data | Quando a OS foi fechada |
    | Tem Histórico? | Sim/Não | Se o cliente tem OS anteriores |
    | Dias desde Última OS | Número | Intervalo em dias (se tem histórico) |
    
    **⚠️ Importante:**
    - Os IDs de produto e defeito são convertidos numericamente (Label Encoding)
    - As features temporais são criadas automaticamente a partir das datas
    - Todas as features numéricas são normalizadas antes da predição
    
    **🔄 Fluxo de Dados:**
    ```
    Dados Brutos → Engenharia de Features → Encoding → Normalização → Modelo → Predição
    ```
    
    Para detalhes técnicos completos, consulte [README.md](file:/README.md).
""")

st.markdown("Última atualização: Junho 2026 | Versão: 1.0")
