import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
import time
from datetime import datetime

# --- CONFIGURAÇÃO DE AMBIENTE ---
st.set_page_config(page_title="Inteligência de Vigilância Digital", layout="wide")
st.title("🛰️ Sistema de Análise Epidemiológica Preditiva")
st.markdown("---")

# --- BIBLIOTECA DE ONTOLOGIA MÉDICA (Doenças e seus Sintomas Relacionados) ---
# O sistema expande a busca para captar a jornada do paciente, não apenas o nome da doença.
BIBLIOTECA_VIGILANCIA = {
    "Dengue": {
        "termos": ["dengue", "dor atrás dos olhos", "manchas vermelhas", "exantema", "plaquetas baixas"],
        "confundidores": ["Gripe", "Zika"],
        "cor": "#e63946"
    },
    "Gripe/Influenza": {
        "termos": ["gripe", "tosse seca", "dor de garganta", "influenza", "calafrios"],
        "confundidores": ["COVID-19", "Resfriado"],
        "cor": "#457b9d"
    },
    "COVID-19": {
        "termos": ["covid", "perda de paladar", "falta de ar", "teste covid", "anosmia"],
        "confundidores": ["Gripe", "Sinusite"],
        "cor": "#1d3557"
    },
    "Doenças Gastrointestinais": {
        "termos": ["diarreia", "vômito", "dor abdominal", "enjoo", "desidratação"],
        "confundidores": ["Intoxicação Alimentar", "Virose"],
        "cor": "#2a9d8f"
    },
    "Saúde Mental (Ansiedade/Pânico)": {
        "termos": ["ansiedade", "crise de pânico", "falta de ar ansiedade", "palpitação", "insônia"],
        "confundidores": ["Problemas Cardíacos", "Estresse"],
        "cor": "#8e44ad"
    }
}

# --- INICIALIZAÇÃO DO MOTOR ---
try:
    pytrends = TrendReq(hl='pt-BR', tz=360)
except Exception as e:
    st.error(f"Falha na conexão com o servidor de dados: {e}")

# --- INTERFACE DE INVESTIGAÇÃO ---
with st.sidebar:
    st.header("🎯 Parâmetros de Investigação")
    doenca_foco = st.selectbox("Selecione o Agravo Alvo:", list(BIBLIOTECA_VIGILANCIA.keys()))
    uf_foco = st.selectbox("Abrangência Geográfica (UF):", ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR", "BR-GO", "BR-CE", "BR-PE"])
    tempo_analise = st.radio("Janela Temporal:", ["Últimos 3 meses", "Últimos 12 meses"])
    janela = 'today 3-m' if tempo_analise == "Últimos 3 meses" else 'today 12-m'

if st.button(f"🔍 EXECUTAR VARREDURA PROFUNDA: {doenca_foco.upper()}"):
    try:
        dados_foco = BIBLIOTECA_VIGILANCIA[doenca_foco]
        termos_expandidos = dados_foco["termos"]
        
        with st.status("Realizando varredura sindrômica e cruzamento de dados...", expanded=True) as status:
            # 1. Coleta de Dados de Sintomas (O sistema busca todos os termos da biblioteca)
            pytrends.build_payload(termos_expandidos, geo=uf_foco, timeframe=janela)
            df_sintomas = pytrends.interest_over_time()
            if not df_sintomas.empty:
                df_sintomas = df_sintomas.drop(columns=['isPartial'], errors='ignore')
            
            # 2. Coleta para Análise de Distorção (Confundidores)
            confundidor = dados_foco["confundidores"][0]
            pytrends.build_payload([doenca_foco, confundidor], geo=uf_foco, timeframe=janela)
            df_distorsao = pytrends.interest_over_time()
            
            # 3. Coleta Geográfica (Mapa Nacional)
            pytrends.build_payload([doenca_foco], geo='BR', timeframe='today 1-m')
            df_mapa = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
            
            status.update(label="Varredura Concluída!", state="complete")

        if not df_sintomas.empty:
            # --- CÁLCULOS ESTATÍSTICOS DE PRECISÃO ---
            # Índice de Convergência: Se os sintomas sobem juntos, a chance de ser real é alta
            correlacao_matriz = df_sintomas.corr()
            indice_convergencia = correlacao_matriz.mean().mean()
            
            # Cálculo de Anomalia (Z-Score)
            media_sintomas = df_sintomas.mean(axis=1)
            z_scores = zscore(media_sintomas)
            ultimo_z = z_scores[-1]
            
            # --- ÍNDICE DE CHANCE REAL (Vero-Score) ---
            # Combina intensidade (Z-Score) com convergência de sintomas
            probabilidade_real = (indice_convergencia * 0.5 + (min(ultimo_z, 3)/3) * 0.5) * 100
            probabilidade_real = max(0, min(100, probabilidade_real))

            # --- DISPLAY DE RESULTADOS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Chance de Surto Real", f"{probabilidade_real:.1f}%")
            col2.metric("Convergência de Sintomas", f"{indice_convergencia:.2f}")
            col3.metric("Intensidade (Z-Score)", f"{ultimo_z:.2f}")

            st.divider()

            # --- ANÁLISE DETALHADA E PARECER ---
            st.subheader("📝 Parecer Técnico de Investigação")
            
            col_p1, col_p2 = st.columns([2, 1])
            
            with col_p1:
                if probabilidade_real > 70 and ultimo_z > 1.5:
                    st.error(f"**ALERTA CRÍTICO:** O agravo '{doenca_foco}' apresenta alta consistência interna. "
                             f"A subida do interesse ({ultimo_z:.2f} desvios padrão) é acompanhada por uma forte convergência "
                             f"dos sintomas sentinelas ({indice_convergencia:.2f}). Esta assinatura digital é característica de surtos biológicos reais.")
                elif probabilidade_real > 40:
                    st.warning(f"**ALERTA MODERADO:** Existe aumento de buscas para '{doenca_foco}', mas a convergência de sintomas é mediana. "
                               "O dado pode estar sofrendo influência de notícias ou campanhas de conscientização.")
                else:
                    st.success("**SITUAÇÃO SOB CONTROLE:** Interesse residual ou flutuação normal de mercado/noticiário.")

            with col_p2:
                # --- ANÁLISE DE DISTORÇÃO (Diagnóstico Diferencial Digital) ---
                val_foco = df_distorsao[doenca_foco].iloc[-1]
                val_conf = df_distorsao[confundidor].iloc[-1]
                
                st.write("**Risco de Distorção Sintomática**")
                if val_conf > val_foco * 0.7:
                    st.info(f"⚠️ **ALTO RISCO DE ERRO:** As buscas por '{confundidor}' estão muito próximas de '{doenca_foco}'. "
                            f"Como estas patologias compartilham sinais clínicos, o aumento detectado pode ser um 'falso positivo' "
                            f"causado por uma epidemia de {confundidor}.")
                else:
                    st.write("✅ **DADOS CONSISTENTES:** A curva desta patologia está isolada de seus principais confundidores clínicos.")

            # --- VISUALIZAÇÕES GRÁFICAS ---
            st.divider()
            tab_mapa, tab_sintomas, tab_distorsao = st.tabs(["🗺️ Mapa Geográfico", "📈 Convergência de Sintomas", "🔄 Análise Comparativa"])

            with tab_mapa:
                st.subheader("Disseminação Espacial (Mês Atual)")
                df_mapa_res = df_mapa.reset_index()
                fig_mapa = px.choropleth(
                    df_mapa_res,
                    geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                    locations='geoName',
                    featureidkey="properties.name",
                    color=df_mapa_res.columns[1],
                    color_continuous_scale="Reds",
                    scope="south america",
                    template="plotly_dark"
                )
                fig_mapa.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig_mapa, use_container_width=True)

            with tab_sintomas:
                st.subheader("Comportamento dos Sintomas Sentinelas")
                st.line_chart(df_sintomas)
                st.caption("A proximidade e sincronia entre as linhas indicam a validade epidemiológica do surto.")

            with tab_distorsao:
                st.subheader(f"Diferencial: {doenca_foco} vs {confundidor}")
                st.line_chart(df_distorsao[[doenca_foco, confundidor]])
                st.write("Se a linha do confundidor estiver acima ou colada na linha alvo, a especificidade do dado digital é baixa.")

    except Exception as e:
        st.error(f"Erro na varredura: {e}. O Google pode ter limitado o acesso. Aguarde alguns minutos.")
