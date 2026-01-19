"""
SISTEMA SENTINELA DE VIGILÂNCIA EPIDEMIOLÓGICA INTEGRADA (SVEI) - V18.0 PEER-REVIEWED
-------------------------------------------------------------------------------------
Arquitetura: Modular Orientada a Objetos com Correções de Viés e Rigor Estatístico.
Autor: Gemini AI (Refinado após revisão de pares)

CHANGELOG V18 (FINAL):
- [FIX] Viz: Gráfico usa 'Layered Traces' para garantir preenchimento de alerta correto (sem artefatos).
- [SCI] Stats: Baseline MAD com 'Shift(1)' para evitar contaminação pelo dado presente (Look-ahead bias).
- [SCI] Text: Remoção de alegações de causalidade; adoção de "Associação Temporal".
- [SCI] Metric: KSU definido explicitamente como Proxy de Demanda Relativa.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from scipy.stats import pearsonr
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAÇÃO E DADOS DE REFERÊNCIA
# ==============================================================================

st.set_page_config(
    page_title="SVEI Sentinel v18",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

# Dados Demográficos (IBGE)
POPULACAO_UF = {
    'BR-SP': 44411238, 'BR-MG': 21411923, 'BR-RJ': 17463349, 'BR-BA': 14985284,
    'BR-PR': 11597484, 'BR-RS': 11466630, 'BR-PE': 9674793, 'BR-CE': 9240580,
    'BR-PA': 8777124, 'BR-SC': 7338473, 'BR-GO': 7206589, 'BR-MA': 7153262,
    'BR-AM': 4269995, 'BR-ES': 4108508, 'BR-PB': 4059905, 'BR-MT': 3567234,
    'BR-RN': 3560903, 'BR-AL': 3365351, 'BR-PI': 3289290, 'BR-DF': 3094325,
    'BR-MS': 2839188, 'BR-SE': 2338474, 'BR-RO': 1815278, 'BR-TO': 1607363,
    'BR-AC': 906876, 'BR-AP': 877613, 'BR-RR': 652713
}

# Estimativa de Penetração de Internet (PNAD)
PENETRACAO_INTERNET = {
    'BR-DF': 0.92, 'BR-SP': 0.88, 'BR-RJ': 0.86, 'BR-SC': 0.85,
    'BR-PR': 0.84, 'BR-RS': 0.83, 'BR-GO': 0.82, 'BR-MS': 0.81,
    'BR-MG': 0.80, 'BR-ES': 0.79, 'BR-MT': 0.78, 'BR-RO': 0.75,
    'BR-TO': 0.72, 'BR-BA': 0.70, 'BR-SE': 0.69, 'BR-RN': 0.68,
    'BR-CE': 0.67, 'BR-PE': 0.66, 'BR-PB': 0.65, 'BR-PA': 0.62,
    'BR-AM': 0.61, 'BR-AP': 0.60, 'BR-RR': 0.59, 'BR-MA': 0.58,
    'BR-AL': 0.57, 'BR-PI': 0.56, 'BR-AC': 0.55
}

# Cores
C_ALVO = '#2980b9'
C_LIMIAR = '#c0392b'
C_FILL = 'rgba(231, 76, 60, 0.3)'

# ==============================================================================
# 2. MÓDULO DE SIMULAÇÃO (FAILOVER SYSTEM)
# ==============================================================================
class MockDataGenerator:
    """Gera dados sintéticos com seed fixa para reprodutibilidade."""
    @staticmethod
    def gerar_curva_epidemiologica(dias=90, intensidade=1.0, seed=42):
        np.random.seed(seed)
        x = np.linspace(0, 10, dias)
        surto = 100 * (1 / (1 + np.exp(-(x - 5)*2))) * np.exp(-(x - 5)*0.2)
        baseline = 10 + 5 * np.sin(x)
        ruido = np.random.normal(0, 3, dias)
        y = (baseline + (surto * intensidade) + ruido)
        return np.clip(y, 0, 100)

    @staticmethod
    def criar_dataset_simulado(termos):
        dates = pd.date_range(end=datetime.today(), periods=90)
        data = {}
        data[termos[0]] = MockDataGenerator.gerar_curva_epidemiologica(seed=42)
        raw_sintoma = MockDataGenerator.gerar_curva_epidemiologica(seed=42)
        data[termos[1]] = np.roll(raw_sintoma, -5) * 0.9 
        raw_remedio = MockDataGenerator.gerar_curva_epidemiologica(seed=42)
        data[termos[2]] = np.roll(raw_remedio, 2) * 0.7
        np.random.seed(99)
        data[termos[3]] = np.clip(data[termos[0]] * np.random.uniform(0.5, 1.5, 90), 0, 100)
        np.random.seed(101)
        data[termos[4]] = np.random.normal(30, 5, 90)
        return pd.DataFrame(data, index=dates)

# ==============================================================================
# 3. MÓDULO DE CONEXÃO
# ==============================================================================
class TrendMiningAgent:
    def __init__(self):
        self.hl = 'pt-BR'
        self.tz = 180 # UTC-3

    @st.cache_data(ttl=3600, show_spinner=False)
    def buscar_dados(_self, termos, geo, timeframe):
        pytrends = TrendReq(hl=_self.hl, tz=_self.tz)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pytrends.build_payload(termos, geo=geo, timeframe=timeframe)
                df = pytrends.interest_over_time()
                if not df.empty:
                    return df.drop(columns=['isPartial'], errors='ignore'), False
            except Exception as e:
                time.sleep(2 ** attempt)
        return MockDataGenerator.criar_dataset_simulado(termos), True

# ==============================================================================
# 4. MÓDULO MATEMÁTICO (MATH ENGINE)
# ==============================================================================
class EpidemiologicalMath:
    
    @staticmethod
    def definir_janela_adaptativa(timeframe):
        if "1-m" in timeframe: return 5
        if "3-m" in timeframe: return 7
        if "12-m" in timeframe: return 21
        return 7

    @staticmethod
    def aplicar_media_movel_retrospectiva(df, janela):
        """Suavização 'Honesta': center=False"""
        df_smooth = df.copy()
        for col in df.columns:
            df_smooth[f'{col}_smooth'] = df[col].rolling(window=janela, center=False, min_periods=1).mean()
        return df_smooth

    @staticmethod
    def calcular_limiar_robusto_mad(serie, janela):
        """
        CORREÇÃO CRÍTICA V18: SHIFT(1)
        O baseline é calculado com dados até ONTEM. O dado de HOJE não influencia o limite de HOJE.
        Isso evita que um surto súbito 'suba a régua' instantaneamente e mascare o alerta.
        """
        # Shiftamos a série 1 dia para trás antes de calcular a mediana/MAD
        serie_shifted = serie.shift(1)
        
        # Agora calculamos sobre a série deslocada
        roll_median = serie_shifted.rolling(window=janela*2, center=False, min_periods=1).median()
        
        roll_mad = serie_shifted.rolling(window=janela*2, center=False, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x)))
        )
        
        # Limiar = Mediana + 3 * MAD (Equivalente a 3 Sigmas)
        # Usamos 1.4826 como fator de consistência para distribuição normal
        threshold = roll_median + (3 * roll_mad * 1.4826)
        
        return threshold, roll_median

    @staticmethod
    def calcular_detrended_lag_significancia(alvo, preditor, max_lag=14):
        """Lead-Time Detrended com P-Valor"""
        d_alvo = alvo.diff().fillna(0)
        d_preditor = preditor.diff().fillna(0)
        
        best_lag = 0
        best_corr = -1.0
        best_p_value = 1.0
        
        for lag in range(1, max_lag + 1):
            d_pred_shifted = d_preditor.shift(lag).fillna(0)
            try:
                corr, p_val = pearsonr(d_alvo, d_pred_shifted)
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                    best_lag = lag
                    best_p_value = p_val
            except:
                continue
        return best_lag, best_corr, best_p_value

    @staticmethod
    def calcular_vero_index_robusto(clinico, ruido, controle):
        return clinico / ((ruido * 0.5) + (controle * 0.5) + 0.1)

# ==============================================================================
# 5. MÓDULO DEMOGRÁFICO
# ==============================================================================
class DemographicAdjuster:
    @staticmethod
    def calcular_impacto_proxy(valor_relativo, uf_code):
        """Impacto normalizado KSU (Proxy de Demanda)"""
        pop = POPULACAO_UF.get(uf_code, 1000000)
        penetracao = PENETRACAO_INTERNET.get(uf_code, 0.70)
        pop_conectada = pop * penetracao
        # Normalização por 100k usuários conectados
        score_ksu = (valor_relativo / 100) * (pop_conectada / 100000)
        return score_ksu, pop_conectada

# ==============================================================================
# 6. FRONTEND (STREAMLIT)
# ==============================================================================

# --- Sidebar ---
st.sidebar.title("🛡️ SVEI Sentinel")
st.sidebar.caption("v18.0 Academic Peer-Review")
st.sidebar.markdown("---")

raw_doenca = st.sidebar.text_input("Termo Sentinela:", value="Dengue")
input_doenca = raw_doenca.strip()
input_uf = st.sidebar.selectbox("Jurisdição:", options=list(POPULACAO_UF.keys()))

st.sidebar.markdown("### ⚙️ Parâmetros")
janela_analise = st.sidebar.selectbox("Timeframe:", ["today 3-m", "today 1-m", "today 12-m"])

# --- Main Logic ---
st.title(f"Monitor Sentinela: {input_doenca.upper()}")

if st.button("🔎 INICIAR PROTOCOLO SENTINELA", type="primary"):
    
    miner = TrendMiningAgent()
    math = EpidemiologicalMath()
    demo = DemographicAdjuster()
    
    termos = [
        input_doenca,                        
        f"sintomas {input_doenca}",          
        f"remedio {input_doenca}",           
        f"noticia {input_doenca}",           
        "previsão do tempo"                  
    ]
    
    with st.spinner("Processando dados e inferências estatísticas..."):
        # 1. Busca
        df_raw, is_simulated = miner.buscar_dados(termos, input_uf, janela_analise)
        if is_simulated:
            st.warning("⚠️ FAILOVER ATIVO: Usando dados sintéticos para demonstração.")

        # 2. Matemáticas
        win_size = math.definir_janela_adaptativa(janela_analise)
        df_smooth = math.aplicar_media_movel_retrospectiva(df_raw, janela=win_size)
        
        c_alvo = f"{termos[0]}_smooth"
        c_clinico = f"{termos[1]}_smooth"
        
        # Baseline Robusto (Shifted)
        threshold, trend = math.calcular_limiar_robusto_mad(df_raw[termos[0]], win_size)
        df_smooth['threshold'] = threshold
        
        # Lag
        lag_dias, lag_corr, p_val = math.calcular_detrended_lag_significancia(df_raw[termos[0]], df_raw[termos[1]])
        
        # Vero-Index
        c_ruido = f"{termos[3]}_smooth"
        c_controle = f"{termos[4]}_smooth"
        vero_idx = math.calcular_vero_index_robusto(
            df_smooth[c_clinico].iloc[-1],
            df_smooth[c_ruido].iloc[-1],
            df_smooth[c_controle].iloc[-1]
        )
        
        # Impacto KSU
        val_google = df_smooth[c_alvo].iloc[-1]
        impacto_ksu, pop_con = demo.calcular_impacto_proxy(val_google, input_uf)

    # --- RESULTADOS ---
    st.divider()
    
    k1, k2, k3, k4 = st.columns(4)
    status_surto = val_google > threshold.iloc[-1]
    
    k1.metric(
        "Impacto Estimado (Proxy)",
        f"{impacto_ksu:.1f} KSU",
        "Buscas relativas / 100k conectados",
        help="Proxy de demanda proporcional à população conectada da UF."
    )
    
    k2.metric(
        "Status Sentinela",
        "ALERTA" if status_surto else "BASAL",
        f"{(val_google - threshold.iloc[-1]):.1f} pts (vs Limiar)",
        delta_color="inverse"
    )
    
    sig_text = "p<0.05" if p_val < 0.05 else "n.s."
    k3.metric(
        "Lead-Time (Detrended)",
        f"{lag_dias} dias",
        f"Sig: {sig_text}",
        help="Associação temporal baseada na variação diária. Sem ajuste para múltiplos testes."
    )
    
    k4.metric(
        "Vero-Index",
        f"{vero_idx:.2f}",
        "Sinal Puro" if vero_idx > 0.8 else "Ruído",
        help="Razão Sinal Clínico / (Ruído Midiático + Controle)."
    )

    # GRÁFICO CORRIGIDO (PLOTLY LAYERING)
    st.subheader("📉 Monitoramento de Limiar Robusto (MAD Shifted)")
    
    fig = go.Figure()
    
    # Camada 1: Limiar (Base)
    fig.add_trace(go.Scatter(
        x=df_smooth.index, y=df_smooth['threshold'],
        mode='lines', name='Limiar (Mediana + 3 MAD)',
        line=dict(color=C_LIMIAR, dash='dash', width=2)
    ))
    
    # Camada 2: Excesso (Área de Alerta)
    # Lógica: Criamos uma linha que é o MÁXIMO entre o Sinal e o Limiar.
    # Ao preencher "tonexty" (para baixo, até o Limiar), pintamos apenas o excesso.
    y_top = np.maximum(df_smooth[c_alvo], df_smooth['threshold'])
    
    fig.add_trace(go.Scatter(
        x=df_smooth.index, y=y_top,
        mode='lines', line=dict(width=0), # Linha invisível
        fill='tonexty', # Preenche até o Trace anterior (Limiar)
        fillcolor=C_FILL,
        name='Excesso (Sinal > Limiar)',
        hoverinfo='skip'
    ))
    
    # Camada 3: Sinal (Topo)
    fig.add_trace(go.Scatter(
        x=df_smooth.index, y=df_smooth[c_alvo],
        mode='lines', name='Volume Sentinela',
        line=dict(color=C_ALVO, width=3)
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    

    # TRIAGEM
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📋 Parecer de Triagem")
        if status_surto:
            if vero_idx > 1.0 and p_val < 0.05:
                st.error("🚨 **PRIORIDADE 1:** Anomalia estatística confirmada com precedência temporal significativa.")
            else:
                st.warning("⚠️ **PRIORIDADE 2:** Anomalia detectada, mas com sinais mistos (Ruído ou Lag não-significativo).")
        else:
            st.success("✅ **PRIORIDADE 3:** Comportamento dentro da variabilidade esperada (MAD).")
            
    with c2:
        st.markdown("### 📥 Auditoria")
        st.download_button(
            "Baixar Relatório (CSV)",
            df_raw.to_csv().encode('utf-8'),
            f"svei_audit_{datetime.now().date()}.csv"
        )
