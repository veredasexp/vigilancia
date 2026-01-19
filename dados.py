import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import time

# Configuração da página
st.set_page_config(page_title="Vigilância Epidemiológica Pro", layout="wide")

st.title("🛰️ Sistema Nacional de Vigilância Digital")
st.markdown("Análise de tendências baseada em buscas do Google")

# Dicionário de estados brasileiros
estados = {
    "Brasil (Nacional)": "BR",
    "Acre": "BR-AC", "Alagoas": "BR-AL", "Amapá": "BR-AP", "Amazonas": "BR-AM",
    "Bahia": "BR-BA", "Ceará": "BR-CE", "Distrito Federal": "BR-DF", "Espírito Santo": "BR-ES",
    "Goiás": "BR-GO", "Maranhão": "BR-MA", "Mato Grosso": "BR-MT", "Mato Grosso do Sul": "BR-MS",
    "Minas Gerais": "BR-MG", "Pará": "BR-PA", "Paraíba": "BR-PB", "Paraná": "BR-PR",
    "Pernambuco": "BR-PE", "Piauí": "BR-PI", "Rio de Janeiro": "BR-RJ", "Rio Grande do Norte": "BR-RN",
    "Rio Grande do Sul": "BR-RS", "Rondônia": "BR-RO", "Roraima": "BR-RR", "Santa Catarina": "BR-SC",
    "São Paulo": "BR-SP", "Sergipe": "BR-SE", "Tocantins": "BR-TO"
}

# Interface de filtros
col1, col2 = st.columns(2)
with col1:
    sintoma = st.text_input("Sintoma ou Agravo (ex: dengue, diarreia):", "dengue")
with col2:
    uf_selecionada = st.selectbox("Abrangência Geográfica:", list(estados.keys()))

if st.button("📊 GERAR RELATÓRIO DE VIGILÂNCIA"):
    try:
        pytrends = TrendReq(hl='pt-BR', tz=360)
        pytrends.build_payload([sintoma], geo=estados[uf_selecionada], timeframe='today 3-m')
        df = pytrends.interest_over_time()

        if not df.empty:
            # Cálculos de Inteligência
            df['Tendência (Média 7d)'] = df[sintoma].rolling(window=7).mean()
            hoje = df[sintoma].iloc[-1]
            tendencia = df['Tendência (Média 7d)'].iloc[-1]
            media_passada = df['Tendência (Média 7d)'].iloc[-8] if len(df) > 8 else tendencia
            
            # Gráfico Principal
            st.subheader(f"Evolução Temporal: {sintoma} em {uf_selecionada}")
            st.line_chart(df[[sintoma, 'Tendência (Média 7d)']])

            # PARECER TÉCNICO AUTOMÁTICO
            st.markdown("---")
            st.subheader("📝 Parecer Técnico")
            
            c1, c2 = st.columns(2)
            
            # Análise de Intensidade
            with c1:
                st.write("**Intensidade Atual:**")
                if hoje > tendencia * 1.2:
                    st.error(f"ALERTA: O interesse hoje ({hoje}) está significativamente acima da média móvel. Risco de surto detectado.")
                else:
                    st.success("ESTÁVEL: O interesse atual está dentro dos parâmetros normais da semana.")

            # Análise de Tendência
            with c2:
                variacao = ((tendencia - media_passada) / (media_passada + 0.1)) * 100
                st.write("**Evolução Semanal:**")
                if variacao > 10:
                    st.warning(f"ACELERAÇÃO: Aumento de {variacao:.1f}% na tendência em relação à semana anterior.")
                elif variacao < -10:
                    st.info(f"QUEDA: Redução de {abs(variacao):.1f}% na tendência observada.")
                else:
                    st.write("ESTABILIDADE: Não houve variação significativa na última semana.")

            st.info("**Nota Metodológica:** Os dados refletem o volume de buscas no Google, funcionando como um sensor antecipado de casos reais.")

        else:
            st.warning("Dados insuficientes para esta região. Tente um termo mais comum.")
            
    except Exception as e:
        st.error("Ocorreu um erro ou o Google limitou o acesso. Tente novamente em alguns minutos.")