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
# CONFIGURAÇÃO DE INTERFACE E CABEÇALHO CIENTÍFICO
# =================================================================
st.set_page_config(page_title="Vigilância de Alta Precisão", layout="wide")

st.title("🔬 Sistema de Inteligência Epidemiológica Digital")
st.markdown("""
Esta plataforma realiza **Vigilância Baseada em Rumores (VBR)** de alta fidelidade. 
O sistema analisa a consistência interna dos sintomas e a distorção por síndromes sobrepostas para determinar a probabilidade de um surto real.
""")

# =================================================================
# BIBLIOTECA ONTOLÓGICA (DOENÇAS, SINTOMAS E CONFUNDIDORES)
# =================================================================
BIBLIOTECA_VIGILANCIA = {
    "Dengue": {
        "termos": ["dengue", "dor atrás dos olhos", "manchas vermelhas", "febre alta", "plaquetas"],
        "confundidores": ["Gripe", "Zika", "Leptospirose"],
        "descricao": "Arbovirose clássica. A análise foca na convergência da tríade febre-exantema-dor retro-orbital."
    },
    "Gripe / Influenza": {
        "termos": ["gripe", "tosse seca", "dor de garganta", "coriza", "influenza"],
        "confundidores": ["COVID-19", "Resfriado Comum", "Pneumonia"],
        "descricao": "Monitoramento de vírus respiratórios sazonais com foco em sintomas de via aérea superior."
    },
    "COVID-19": {
        "termos": ["covid", "falta de ar", "teste covid", "perda de paladar", "perda de olfato"],
        "confundidores": ["Gripe", "Sinusite", "Ansiedade"],
        "descricao": "Vigilância de SARS-CoV-2 com filtragem por sintomas patognomônicos (anosmia/ageusia)."
    },
    "Doenças Gastrointestinais": {
        "termos": ["diarreia", "vômito", "enjoo", "dor abdominal", "desidratação"],
        "confundidores": ["Intoxicação Alimentar", "Virose", "Cólera"],
        "descricao": "Vigilância de agravos de transmissão hídrica e alimentar."
    }
}

# =================================================================
# MOTOR DE DADOS COM TRATAMENTO DE ERROS E CACHE
# =================================================================
@st.cache_resource
def conectar_google():
    """Conecta à API do Google Trends sem os argumentos legados que causam erro."""
    return TrendReq(hl='pt-BR', tz=360)

@st.cache_data(ttl=3600)
def requisitar_dados(termos, geo, timeframe):
    """Executa a busca com persistência e cache."""
    pytrends = conectar_google()
    try:
        pytrends.build_payload(termos, geo=geo, timeframe=timeframe)
        df = pytrends.interest_over_time()
        if not df.empty:
            return df.drop(columns=['isPartial'], errors='ignore')
        return None
    except Exception as e:
        if "429" in str(e):
            st.error("🚨 Limite de taxa do Google atingido. Aguarde 10 minutos ou tente outra UF.")
        else:
            st.error(f"Erro na requisição: {e}")
        return None

# =================================================================
# INTERFACE DE SELEÇÃO E CONTROLE
# =================================================================
with st.sidebar:
    st.header("🎯 Parâmetros de Investigação")
    doenca_alvo = st.selectbox("Selecione o Agravo Alvo:", list(BIBLIOTECA_VIGILANCIA.keys()))
    uf_alvo = st.sidebar.selectbox("Estado (UF):", ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR", "BR-GO", "BR-CE", "BR-PE"])
    st.markdown("---")
    st.info("O sistema consultará automaticamente todos os sintomas relacionados e comparará com síndromes espelho.")

# =================================================================
# EXECUÇÃO DA ANÁLISE
# =================================================================
if st.button(f"🚀 INICIAR VARREDURA PROFUNDA: {doenca_alvo.upper()}"):
    info = BIBLIOTECA_VIGILANCIA[doenca_alvo]
    termos_investigacao = info["termos"]
    confundidor_principal = info["confundidores"][0]

    with st.status("Executando Protocolo de Inteligência Epidemiológica...", expanded=True) as status:
        # 1. Coleta de Sintomas Sentinelas
        st.write("Analisando convergência de sintomas...")
        df_sintomas = requisitar_dados(termos_investigacao, uf_alvo, 'today 3-m')
        time.sleep(2)
        
        # 2. Coleta de Dados de Distorção (Diferencial Digital)
        st.write("Calculando risco de distorção por síndromes sobrepostas...")
        df_distorsao = requisitar_dados([doenca_alvo, confundidor_principal], uf_alvo, 'today 3-m')
        time.sleep(2)
        
        # 3. Coleta Geográfica Nacional
        st.write("Mapeando disseminação espacial...")
        df_mapa = requisitar_dados([doenca_alvo], 'BR', 'today 1-m')
        
        status.update(label="Análise Concluída!", state="complete")

    if df_sintomas is not None:
        # ---------------------------------------------------------
        # CÁLCULOS ESTATÍSTICOS DE VERACIDADE
        # ---------------------------------------------------------
        # Sincronia: Quão juntos os sintomas caminham (Matriz de Correlação)
        matriz_corr = df_sintomas.corr()
        convergencia = matriz_corr.mean().mean()
        
        # Intensidade: Z-Score para detectar anomalia estatística
        media_temporal = df_sintomas.mean(axis=1)
        scores_z = zscore(media_temporal)
        ultimo_z = scores_z[-1]
        
        # Vero-Score (Chance Real): Pesa Sincronia (40%) e Intensidade (60%)
        chance_real = (convergencia * 0.4 + (min(ultimo_z, 3)/3) * 0.6) * 100
        chance_real = max(0, min(100, chance_real))

        # ---------------------------------------------------------
        # PAINEL DE RESULTADOS
        # ---------------------------------------------------------
        st.header(f"📊 Relatório de Investigação: {doenca_alvo}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Chance de Surto Real", f"{chance_real:.1f}%")
        c2.metric("Convergência de Sintomas", f"{convergencia:.2f}")
        c3.metric("Intensidade (Z-Score)", f"{ultimo_z:.2f}")

        st.divider()

        # ---------------------------------------------------------
        # ANÁLISE DE DISTORÇÃO E PARECER TÉCNICO
        # ---------------------------------------------------------
        col_txt, col_graph = st.columns([2, 1])
        
        with col_txt:
            st.subheader("📝 Parecer Técnico")
            # Lógica de Diagnóstico de Distorção
            if df_distorsao is not None:
                val_alvo = df_distorsao[doenca_alvo].iloc[-1]
                val_conf = df_distorsao[confundidor_principal].iloc[-1]
                
                if val_conf > val_alvo * 0.7:
                    st.error(f"⚠️ **ALERTA DE DISTORÇÃO:** As buscas por '{confundidor_principal}' estão muito elevadas. "
                             f"Dado que ambas compartilham sintomas, o aumento em {doenca_alvo} pode ser um 'falso positivo' "
                             f"ou estar mascarado por um surto paralelo de {confundidor_principal}.")
                else:
                    st.success(f"**DADOS CONSISTENTES:** O sinal para {doenca_alvo} é específico e apresenta baixo ruído de doenças espelho.")
            
            if chance_real > 70:
                st.markdown(f"**Conclusão:** Há evidências digitais robustas de um surto de **{doenca_alvo}**. "
                            f"O aumento de buscas é suportado por sintomas clínicos sincronizados.")
            else:
                st.markdown("**Conclusão:** O interesse atual parece ser movido por curiosidade informacional ou notícias, sem base sintomática convergente.")

        with col_graph:
            if df_distorsao is not None:
                st.write("**Diferencial Digital**")
                st.line_chart(df_distorsao[[doenca_alvo, confundidor_principal]])

        # ---------------------------------------------------------
        # VISUALIZAÇÕES GRÁFICAS AVANÇADAS
        # ---------------------------------------------------------
        st.divider()
        tab_mapa, tab_sintomas = st.tabs(["🗺️ Mapa Coroplético Nacional", "📈 Convergência de Sintomas Sentinelas"])

        with tab_mapa:
            if df_mapa is not None:
                st.subheader("Disseminação Geográfica (Interesse por Estado)")
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

        with tab_sintomas:
            st.subheader("Sincronia Temporal dos Sinais Clínicos")
            st.line_chart(df_sintomas)
            st.caption("A proximidade entre as curvas de sintomas diferentes valida a ocorrência de casos clínicos reais.")

    else:
        st.warning("Não foi possível processar a varredura. Verifique a conexão ou tente novamente mais tarde.")

# --- FOOTER ---
st.divider()
st.caption(f"Plataforma de Vigilância Digital v8.0 | Análise gerada em: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
