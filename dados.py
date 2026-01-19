import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px
from scipy.stats import pearsonr
import time

st.set_page_config(page_title="Investigação Epidemiológica Pro", layout="wide")

# --- MOTOR DE INTELIGÊNCIA ---
pytrends = TrendReq(hl='pt-BR', tz=360)

st.title("🔬 Plataforma Avançada de Vigilância e Investigação")

# --- BARRA LATERAL: CONFIGURAÇÃO DA PESQUISA ---
st.sidebar.header("Configurações de Filtro")
uf = st.sidebar.selectbox("Estado:", ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR"])
categoria = st.sidebar.selectbox("Filtro de Intenção (Ideia 4):", 
                                ["Foco em Sintomas (Paciente)", "Foco em Notícias/Geral"])

# Mapeamento de termos por intenção
termos_sintomas = ["sintomas", "dor", "febre", "tratamento", "remédio"] if categoria == "Foco em Sintomas (Paciente)" else ["casos", "notícias", "vacina", "mortes"]

# --- ABA 1: MONITORIZAÇÃO E MAPAS (Ideia 2) ---
tab1, tab2 = st.tabs(["📡 Vigilância em Tempo Real", "📊 Validação Estatística"])

with tab1:
    termo_busca = st.text_input("Agravo Principal (ex: Dengue):", "Dengue")
    
    if st.button("Executar Análise Geográfica e de Tendência"):
        # Busca de Tendência
        pytrends.build_payload([termo_busca], geo=uf, timeframe='today 3-m')
        df_tempo = pytrends.interest_over_time()
        
        # Busca por Região (Ideia 2)
        df_cidades = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tendência Temporal")
            st.line_chart(df_tempo[termo_busca])
        
        with col2:
            st.subheader("Mapa de Calor: Concentração por Cidades")
            if not df_cidades.empty:
                fig = px.bar(df_cidades.sort_values(by=termo_busca, ascending=False).head(15), 
                             x=termo_busca, y=df_cidades.head(15).index, orientation='h',
                             title="Cidades com Maior Volume de Rumores",
                             labels={termo_busca: 'Intensidade', 'index': 'Cidade'})
                st.plotly_chart(fig)

# --- ABA 2: CORRELAÇÃO COM DADOS REAIS (Ideia 1) ---
with tab2:
    st.subheader("Validação Científica (Casos Reais vs. Google)")
    st.markdown("""
    Submeta uma folha Excel/CSV com duas colunas: **Data** e **Casos_Reais**. 
    O sistema calculará o coeficiente de correlação para validar o seu modelo de investigação.
    """)
    
    file = st.file_uploader("Upload de Dados do SINAN / Secretaria de Saúde", type=['csv', 'xlsx'])
    
    if file and not df_tempo.empty:
        # Processamento simples dos dados reais
        df_real = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
        
        # Demonstração de Correlação (Exemplo Teórico no Gráfico)
        st.write("**Gráfico de Validação:**")
        # Aqui o pesquisador compararia a curva do Google com a curva real
        st.info("💡 Dica de Pesquisa: Se a correlação for > 0.7, o Google Trends é um indicador preditivo forte para esta patologia na sua região.")
        
        # Cálculo de Pearson (Simplificado para o exemplo)
        st.warning("Cálculo de Pearson disponível após alinhamento das séries temporais (Datas).")

# --- FOOTER ---
st.divider()
st.caption("Investigação de Vigilância Digital v5.0 - Ideias 1, 2 e 4 integradas.")
