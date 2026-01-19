import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr
import time
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA INTERFACE ---
st.set_page_config(page_title="Vigilância Epidemiológica Avançada", layout="wide")
st.title("🔬 Plataforma de Vigilância Digital e Validação Epidemiológica")
st.markdown("""
Esta plataforma realiza a varredura proativa de síndromes e valida a eficácia dos rumores digitais 
cruzando-os com dados reais de internações.
""")

# --- MOTOR DE BUSCA (BACKEND) ---
try:
    pytrends = TrendReq(hl='pt-BR', tz=360)
except:
    st.error("Erro ao conectar ao Google. Tente novamente em instantes.")

# --- DICIONÁRIO DE VIGILÂNCIA (SINDROMES) ---
SINDROMES = {
    "Arboviroses (Dengue/Zika)": ["dengue", "sintomas dengue", "dor atrás dos olhos"],
    "Síndrome Respiratória": ["gripe", "falta de ar", "tosse seca", "influenza"],
    "Síndrome Gastrointestinal": ["diarreia", "vômito", "enjoo", "dor abdominal"],
    "Doenças Exantemáticas": ["manchas vermelhas", "sarampo", "rubéola"]
}

# --- SIDEBAR: INPUT DE DADOS REAIS ---
st.sidebar.header("📂 Validação de Dados Reais")
st.sidebar.markdown("Para calcular a correlação, suba uma planilha com as colunas **'Data'** e **'Internacoes'**.")
arquivo_real = st.sidebar.file_uploader("Upload de dados do SINAN/Hospitais", type=['csv', 'xlsx'])

# --- ABA PRINCIPAL: VARREDURA ---
if st.button("🚀 INICIAR VARREDURA INTEGRAL E ANÁLISE DE CORRELAÇÃO"):
    resultados_globais = []
    
    with st.status("Processando inteligência de dados...", expanded=True) as status:
        for nome_s, termos in SINDROMES.items():
            st.write(f"Analisando: {nome_s}...")
            
            # 1. Coleta Temporal (Últimos 90 dias)
            pytrends.build_payload(termos, geo='BR-MS', timeframe='today 3-m')
            df_trends = pytrends.interest_over_time()
            
            # 2. Coleta Regional (Para o Mapa de Calor)
            pytrends.build_payload([termos[0]], geo='BR', timeframe='today 1-m')
            df_regiao = pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
            
            if not df_trends.empty:
                # Processamento de Médias
                df_trends['media_sindrome'] = df_trends[termos].mean(axis=1)
                hoje = df_trends['media_sindrome'].iloc[-1]
                media_historica = df_trends['media_sindrome'].mean()
                desvio_padrao = df_trends['media_sindrome'].std()
                z_score = (hoje - media_historica) / desvio_padrao if desvio_padrao > 0 else 0
                
                resultados_globais.append({
                    "nome": nome_s,
                    "z_score": z_score,
                    "hoje": hoje,
                    "df": df_trends,
                    "mapa": df_regiao
                })
            time.sleep(1.5) # Proteção de taxa de acesso
        status.update(label="Análise Concluída!", state="complete")

    if resultados_globais:
        # --- IDENTIFICAÇÃO DO AGRAVO PRIORITÁRIO ---
        resultados_globais.sort(key=lambda x: x['z_score'], reverse=True)
        critico = resultados_globais[0]

        # --- SEÇÃO 1: PARECER TÉCNICO DETALHADO ---
        st.header("📝 Parecer Analítico de Vigilância")
        col_txt, col_metric = st.columns([3, 1])
        
        with col_txt:
            interpretação = "estável" if critico['z_score'] < 1.5 else "em alerta moderado" if critico['z_score'] < 2.5 else "em estado crítico de surto"
            st.markdown(f"""
            O sistema realizou a varredura em 4 grandes grupos sindrômicos. O grupo com maior desvio detectado foi **{critico['nome']}**. 
            
            **Análise Estatística:** O valor atual apresenta um **Z-Score de {critico['z_score']:.2f}**. Na epidemiologia digital, valores acima de 2.0 indicam que o volume de buscas rompeu o canal endêmico histórico. 
            Este aumento sugere uma circulação viral ativa no estado, precedendo o pico de notificações oficiais em aproximadamente 7 a 14 dias.
            """)
        
        with col_metric:
            st.metric("Índice de Anomalia", f"{critico['z_score']:.2f}", delta="Crítico" if critico['z_score'] > 2 else "Normal")

        # --- SEÇÃO 2: MAPA DE CALOR (MAPA REAL POR ESTADO) ---
        st.subheader("🗺️ Disseminação Geográfica Nacional")
        df_mapa_res = critico['mapa'].reset_index()
        
        fig_mapa = px.choropleth(
            df_mapa_res,
            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
            locations='geoName',
            featureidkey="properties.name",
            color=df_mapa_res.columns[1],
            color_continuous_scale="Reds",
            scope="south america",
            labels={'geoName': 'Estado', df_mapa_res.columns[1]: 'Intensidade'}
        )
        fig_mapa.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_mapa, use_container_width=True)
        

        # --- SEÇÃO 3: VALIDAÇÃO POR CORRELAÇÃO (DADOS REAIS) ---
        st.divider()
        st.header("📊 Validação Científica: Rumores vs. Internações")
        
        if arquivo_real:
            # Processamento da Planilha
            df_interno = pd.read_csv(arquivo_real) if arquivo_real.name.endswith('csv') else pd.read_excel(arquivo_real)
            df_interno['Data'] = pd.to_datetime(df_interno['Data'])
            
            # Alinhamento das séries
            df_google = critico['df'].reset_index()
            df_google['date'] = pd.to_datetime(df_google['date'])
            
            df_merge = pd.merge(df_google, df_interno, left_on='date', right_on='Data')
            
            if not df_merge.empty:
                # Cálculo de Pearson
                coef_p, p_valor = pearsonr(df_merge['media_sindrome'], df_merge['Internacoes'])
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write("### Coeficiente de Pearson")
                    st.title(f"R = {coef_p:.3f}")
                    if coef_p > 0.7:
                        st.success("✅ **Correlação Forte:** O Google Trends é um preditor confiável para internações neste agravo.")
                    else:
                        st.warning("⚠️ **Correlação Fraca:** Os dados digitais e hospitalares não estão sincronizados.")
                
                with c2:
                    # Gráfico de Duplo Eixo
                    fig_dual = go.Figure()
                    fig_dual.add_trace(go.Scatter(x=df_merge['date'], y=df_merge['media_sindrome'], name="Buscas Google", line=dict(color='blue')))
                    fig_dual.add_trace(go.Scatter(x=df_merge['date'], y=df_merge['Internacoes'], name="Internações Reais", line=dict(color='red'), yaxis="y2"))
                    
                    fig_dual.update_layout(
                        title="Sincronia Temporal: Rumores vs. Fatos",
                        yaxis=dict(title="Volume de Buscas"),
                        yaxis2=dict(title="Nº Internações", overlaying="y", side="right")
                    )
                    st.plotly_chart(fig_dual, use_container_width=True)
            else:
                st.error("As datas da planilha não coincidem com os dados capturados do Google.")
        else:
            st.info("Suba uma planilha de internações na barra lateral para ver a validação estatística aqui.")

        # --- SEÇÃO 4: COMPARATIVO GERAL ---
        st.subheader("📈 Monitoramento Comparativo de Síndromes")
        df_all = pd.DataFrame({r['nome']: r['df']['media_sindrome'] for r in resultados_globais})
        st.line_chart(df_all)
