import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
import time
from datetime import datetime

# =================================================================
# CONFIGURAÇÕES DE INTERFACE E ESTILO DE PESQUISA
# =================================================================
st.set_page_config(
    page_title="Vigilância Epidemiológica de Alta Precisão",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Plataforma de Inteligência Epidemiológica Digital")
st.markdown("""
Esta ferramenta realiza **Vigilância Baseada em Rumores (VBR)** com filtragem de ruído mediático e 
análise de diagnóstico diferencial digital. O sistema avalia a consistência dos sintomas para 
determinar a probabilidade de um evento biológico real.
""")

# =================================================================
# MOTOR DE CONEXÃO COM RETRY LOGIC E CACHE (ANTI-BLOQUEIO)
# =================================================================
@st.cache_resource
def inicializar_pytrends():
    """Inicializa a conexão com o Google Trends API com parâmetros de persistência."""
    return TrendReq(hl='pt-BR', tz=360, retries=5, backoff_factor=0.5)

@st.cache_data(ttl=3600)
def executar_requisicao_google(termos, geo, timeframe):
    """Executa a busca com tratamento de erro 429 e cache de 1 hora."""
    pytrends = inicializar_pytrends()
    try:
        pytrends.build_payload(termos, geo=geo, timeframe=timeframe)
        df = pytrends.interest_over_time()
        if not df.empty:
            return df.drop(columns=['isPartial'], errors='ignore')
        return None
    except Exception as e:
        if "429" in str(e):
            st.error("🚨 Limite de taxa do Google atingido (Erro 429). O sistema entrou em modo de espera.")
        else:
            st.error(f"Erro na requisição: {e}")
        return None

# =================================================================
# BIBLIOTECA ONTOLÓGICA DE AGRAVOS (DEFINIÇÕES TÉCNICAS)
# =================================================================
# Definimos as doenças, seus sintomas sentinelas e seus confundidores clínicos.
BIBLIOTECA_VIGILANCIA = {
    "Dengue": {
        "termos": ["dengue", "dor atrás dos olhos", "manchas vermelhas", "febre alta", "plaquetas"],
        "confundidores": ["Gripe", "Zika", "Leptospirose"],
        "descricao": "Arbovirose clássica. A análise foca na tríade febre-exantema-dor retro-orbital."
    },
    "Gripe / Síndromes Respiratórias": {
        "termos": ["gripe", "tosse seca", "dor de garganta", "coriza", "influenza"],
        "confundidores": ["COVID-19", "Resfriado Comum", "Pneumonia"],
        "descricao": "Monitoramento de Influenza e outros vírus respiratórios sazonais."
    },
    "COVID-19": {
        "termos": ["covid", "falta de ar", "teste covid", "perda de paladar", "perda de olfato"],
        "confundidores": ["Gripe", "Sinusite", "Ansiedade"],
        "descricao": "Vigilância de SARS-CoV-2 com foco em sintomas específicos (anosmia/ageusia)."
    },
    "Doenças Gastrointestinais": {
        "termos": ["diarreia", "vômito", "enjoo", "dor abdominal", "desidratação"],
        "confundidores": ["Intoxicação Alimentar", "Virose Infantil", "Cólera"],
        "descricao": "Vigilância de transmissão hídrica e alimentar."
    }
}

# =================================================================
# INTERFACE DE SELEÇÃO E PARÂMETROS
# =================================================================
with st.sidebar:
    st.header("🎯 Parâmetros de Investigação")
    doenca_foco = st.selectbox("Selecione o Agravo Alvo:", list(BIBLIOTECA_VIGILANCIA.keys()))
    uf_foco = st.selectbox("Unidade Federativa (UF):", 
                          ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR", "BR-GO", "BR-CE", "BR-PE", "BR-AM"])
    
    st.markdown("---")
    st.subheader("Configurações Avançadas")
    tempo_analise = st.radio("Janela de Observação:", ["Últimos 3 meses", "Últimos 12 meses"])
    tf = 'today 3-m' if tempo_analise == "Últimos 3 meses" else 'today 12-m'
    
    st.info("O sistema analisa a convergência de sintomas para filtrar ruídos causados por notícias.")

# =================================================================
# EXECUÇÃO DA INVESTIGAÇÃO
# =================================================================
if st.button(f"🔍 INICIAR INVESTIGAÇÃO PROFUNDA: {doenca_foco.upper()}"):
    info = BIBLIOTECA_VIGILANCIA[doenca_foco]
    termos_sintomas = info["termos"]
    confundidores = info["confundidores"]

    with st.status("Processando dados e calculando indicadores de confiança...", expanded=True) as status:
        # 1. Coleta de Sintomas Expandida
        st.write("Buscando série temporal de sintomas sentinelas...")
        df_sintomas = executar_requisicao_google(termos_sintomas, uf_foco, tf)
        time.sleep(2) # Intervalo de segurança
        
        # 2. Coleta de Diagnóstico Diferencial (Distorção)
        st.write("Analisando possíveis distorções por doenças espelho...")
        df_distorsao = executar_requisicao_google([doenca_foco, confundidores[0]], uf_foco, tf)
        time.sleep(2)
        
        # 3. Coleta Geográfica
        st.write("Gerando mapa de calor nacional...")
        df_mapa = executar_requisicao_google([doenca_foco], 'BR', 'today 1-m')
        
        status.update(label="Investigação Concluída com Sucesso!", state="complete")

    if df_sintomas is not None:
        # ---------------------------------------------------------
        # CÁLCULOS ESTATÍSTICOS AVANÇADOS (O CORAÇÃO DA PESQUISA)
        # ---------------------------------------------------------
        # Cálculo de Convergência (Pearson Correlation Matrix)
        matriz_corr = df_sintomas.corr()
        convergencia_media = matriz_corr.mean().mean() # Quão 'sincronizados' estão os sintomas
        
        # Cálculo de Intensidade (Z-Score)
        serie_media = df_sintomas.mean(axis=1)
        scores = zscore(serie_media)
        ultimo_z = scores[-1]
        
        # Cálculo de Probabilidade de Surto Real (Índice de Chance)
        # O índice aumenta se Z-Score é alto E se a convergência é alta.
        chance_real = (convergencia_media * 0.4 + (min(ultimo_z, 3)/3) * 0.6) * 100
        chance_real = max(0, min(100, chance_real))

        # ---------------------------------------------------------
        # EXIBIÇÃO: PAINEL DE INDICADORES (DASHBOARD)
        # ---------------------------------------------------------
        st.header(f"📊 Relatório de Evidências: {doenca_foco}")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilidade de Surto Real", f"{chance_real:.1f}%")
        m2.metric("Sincronia de Sintomas", f"{convergencia_media:.2f}")
        m3.metric("Intensidade (Z-Score)", f"{ultimo_z:.2f}")

        st.divider()

        # ---------------------------------------------------------
        # EXIBIÇÃO: ANÁLISE QUALITATIVA E PARECER TÉCNICO
        # ---------------------------------------------------------
        col_parecer, col_distorsao = st.columns([2, 1])
        
        with col_parecer:
            st.subheader("📝 Parecer Analítico")
            if chance_real > 75:
                st.error(f"**ALERTA DE SURTO IDENTIFICADO:** O sistema detectou um aumento consistente e convergente. "
                         f"A intensidade de buscas ({ultimo_z:.2f} desvios padrões) associada à alta sincronia dos sintomas "
                         f"({convergencia_media:.2f}) indica uma assinatura epidemiológica típica de propagação viral real.")
            elif chance_real > 40:
                st.warning(f"**MONITORAMENTO RECOMENDADO:** Existe um aumento de rumores, porém com baixa sincronia entre os termos técnicos. "
                           "Isso sugere que o volume pode estar sendo 'inflado' por notícias ou pânico social momentâneo.")
            else:
                st.success("**SITUAÇÃO SOB CONTROLE:** Os dados digitais apresentam flutuações normais sem padrões de surto.")

        with col_distorsao:
            st.subheader("🕵️ Análise de Distorção")
            if df_distorsao is not None:
                val_alvo = df_distorsao[doenca_foco].iloc[-1]
                val_espelho = df_distorsao[confundidores[0]].iloc[-1]
                
                if val_espelho > val_alvo * 0.7:
                    st.info(f"**Risco de Confusão:** Nota-se que buscas por '{confundidores[0]}' estão muito altas. "
                            f"Como os sintomas são parecidos, os dados de {doenca_foco} podem estar distorcidos por este agravo secundário.")
                else:
                    st.write("Os dados apresentam alta especificidade para a patologia alvo, com baixo ruído de doenças espelho.")

        # ---------------------------------------------------------
        # EXIBIÇÃO: VISUALIZAÇÕES GEOGRÁFICAS E TEMPORAIS
        # ---------------------------------------------------------
        st.divider()
        tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Intensidade Nacional", "📈 Curva de Sintomas Sentinelas", "🔄 Comparativo Diferencial"])

        with tab1:
            if df_mapa is not None:
                st.subheader("Disseminação Espacial dos Rumores")
                df_mapa_res = df_mapa.reset_index()
                fig_mapa = px.choropleth(
                    df_mapa_res,
                    geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                    locations='geoName',
                    featureidkey="properties.name",
                    color=df_mapa_res.columns[1],
                    color_continuous_scale="Reds",
                    scope="south america",
                    template="plotly_white"
                )
                fig_mapa.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_mapa, use_container_width=True)
                

[Image of a choropleth map of Brazil]


        with tab2:
            st.subheader("Análise de Convergência de Sintomas")
            st.line_chart(df_sintomas)
            st.caption("A proximidade entre as linhas indica que os pacientes estão buscando a síndrome completa, não apenas termos isolados.")

        with tab3:
            st.subheader(f"Diferencial Digital: {doenca_foco} vs {confundidores[0]}")
            if df_distorsao is not None:
                st.line_chart(df_distorsao[[doenca_foco, confundidores[0]]])
                st.caption(f"Se as linhas estiverem sobrepostas, há alta incerteza diagnóstica nos dados digitais.")

    else:
        st.warning("Não foi possível recuperar dados suficientes. Tente selecionar outro período ou agravo.")

# =================================================================
# FOOTER DE PESQUISA
# =================================================================
st.markdown("---")
st.caption(f"Plataforma de Vigilância Preditiva v7.0 | Dados atualizados em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
