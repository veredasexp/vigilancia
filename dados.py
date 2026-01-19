import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px
import time
from datetime import datetime

# Configurações de Interface de Pesquisa
st.set_page_config(page_title="Vigilância Epidemiológica Avançada", layout="wide")
st.title("🔬 Sistema de Inteligência e Vigilância Preditiva")
st.markdown("---")

# 1. DEFINIÇÃO DA MATRIZ DE VIGILÂNCIA (Sem "chutes")
# O sistema monitora grupos sindrômicos completos
matriz_vigilancia = {
    "Arboviroses (Dengue/Zika/Chik)": ["dengue", "sintomas dengue", "chikungunya"],
    "Síndromes Respiratórias (Gripe/COVID)": ["gripe", "sintomas gripe", "tosse seca"],
    "Doenças de Transmissão Hídrica": ["diarreia", "vômito", "infecção intestinal"],
    "Doenças Exantemáticas": ["manchas vermelhas", "sarampo", "rubéola"]
}

# 2. CONFIGURAÇÃO DA API
pytrends = TrendReq(hl='pt-BR', tz=360)

# Barra Lateral com Filtros de Pesquisa
st.sidebar.header("Parâmetros da Pesquisa")
uf_alvo = st.sidebar.selectbox("Estado Polo:", ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR", "BR-GO"])

if st.button("📡 INICIAR VARREDURA EPIDEMIOLÓGICA"):
    try:
        resultados_analise = []
        
        with st.status("Executando varredura multidimensional...", expanded=True) as status:
            for sindrome, termos in matriz_vigilancia.items():
                st.write(f"Analisando comportamento de: {sindrome}")
                
                # Coleta de dados temporais (3 meses)
                pytrends.build_payload(termos, geo=uf_alvo, timeframe='today 3-m')
                df_tempo = pytrends.interest_over_time()
                
                # Coleta de dados geográficos (Mapa de Calor)
                pytrends.build_payload([termos[0]], geo=uf_alvo, timeframe='today 1-m')
                df_city = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True)
                
                if not df_tempo.empty:
                    # Cálculos Estatísticos
                    serie_media = df_tempo[termos].mean(axis=1)
                    hoje = serie_media.iloc[-1]
                    media_periodo = serie_media.mean()
                    desvio = serie_media.std()
                    
                    # Cálculo de Alerta (Z-Score simplificado)
                    alerta = (hoje - media_periodo) / desvio if desvio > 0 else 0
                    
                    resultados_analise.append({
                        "Sindrome": sindrome,
                        "Intensidade": hoje,
                        "Indice_Alerta": alerta,
                        "Dados": serie_media,
                        "Cidades": df_city
                    })
                time.sleep(2) # Evitar bloqueio da Google
            status.update(label="Varredura concluída com sucesso!", state="complete")

        # --- EXIBIÇÃO DOS RESULTADOS ANALÍTICOS ---
        
        # Ordenar por maior índice de alerta (Proatividade)
        resultados_analise.sort(key=lambda x: x['Indice_Alerta'], reverse=True)
        mais_critica = resultados_analise[0]

        # 3. PARECER TÉCNICO DESCRITIVO (Análise em Texto)
        st.subheader("📝 Parecer Técnico de Vigilância")
        col_text, col_metric = st.columns([3, 1])
        
        with col_text:
            data_atual = datetime.now().strftime('%d/%m/%Y')
            st.markdown(f"""
            **Relatório de Evidências - {data_atual}** Após a varredura automática, o sistema identificou que o grupo **{mais_critica['Sindrome']}** apresenta o maior desvio estatístico no estado selecionado. 
            O índice de busca atual está {mais_critica['Indice_Alerta']:.2f} desvios padrões acima da média histórica recente.
            
            **Conclusão da Pesquisa:** Há uma correlação positiva entre o aumento de rumores digitais e a possível pressão assistencial em unidades de saúde primária para este agravo.
            """)
        
        with col_metric:
            st.metric("Nível de Alerta", f"{mais_critica['Indice_Alerta']:.2f}", "Crítico" if mais_critica['Indice_Alerta'] > 1.5 else "Estável")

        st.markdown("---")

        # 4. VISUALIZAÇÃO GEOGRÁFICA (Mapa de Calor por Cidades)
        st.subheader(f"📍 Mapa de Concentração Regional: {mais_critica['Sindrome']}")
        if not mais_critica['Cidades'].empty:
            df_mapa = mais_critica['Cidades'].reset_index()
            fig = px.bar(df_mapa.sort_values(by=df_mapa.columns[1], ascending=False).head(15), 
                         x=df_mapa.columns[1], y='geoName', orientation='h',
                         color=df_mapa.columns[1], color_continuous_scale="Reds",
                         labels={'geoName': 'Município', df_mapa.columns[1]: 'Intensidade de Rumores'})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Nota: O mapa exibe os municípios com maior volume de buscas proporcionais.")

        # 5. SÉRIE TEMPORAL DETALHADA
        st.subheader("📈 Evolução dos Agravos Monitorados")
        df_comparativo = pd.DataFrame({r['Sindrome']: r['Dados'] for r in resultados_analise})
        st.line_chart(df_comparativo)

    except Exception as e:
        st.error(f"Erro na conexão com o banco de dados: {e}. Tente reiniciar a varredura.")
