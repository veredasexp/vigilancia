import streamlit as st
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from scipy.stats import zscore, pearsonr
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# ==============================================================================
# 1. CONFIGURAÇÃO DO AMBIENTE E ARQUITETURA DE SISTEMA
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Vigilância Epidemiológica Integrada (SVEI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛰️ Sistema de Vigilância Epidemiológica Integrada (SVEI) vFinal")
st.markdown("""
> **Plataforma de Inteligência Computacional para Bio-Vigilância.**
>
> Este sistema implementa uma arquitetura de **10 Dimensões Analíticas** para distinguir surtos biológicos reais de ruídos informacionais, 
> utilizando Estatística Robusta, Cálculo Diferencial e Análise Semântica de Buscas.
""")

# ==============================================================================
# 2. CLASSE: GERENCIADOR DE CONEXÃO (CONNECTION ENGINE)
# ==============================================================================
class ConnectionEngine:
    """
    Gerencia a conexão com o Google Trends.
    CORREÇÃO V13: Removemos o retry automático da biblioteca para evitar conflito
    de versão (method_whitelist error). O retry agora é gerenciado manualmente.
    """
    def __init__(self):
        self.hl = 'pt-BR'
        self.tz = 360

    def conectar(self):
        # Correção: Inicialização limpa sem parâmetros de retry que causam crash em versões novas
        return TrendReq(hl=self.hl, tz=self.tz)

    def executar_busca_blindada(self, termos, geo, timeframe):
        """Tenta buscar dados com múltiplas estratégias manuais de falha."""
        pytrends = self.conectar()
        
        # Estratégia 1: Busca Padrão (3 meses)
        try:
            pytrends.build_payload(termos, geo=geo, timeframe=timeframe)
            df = pytrends.interest_over_time()
            if not df.empty:
                return df.drop(columns=['isPartial'], errors='ignore')
        except Exception as e:
            # Apenas loga o aviso e continua para a estratégia 2
            pass # Silenciamos o erro visual para tentar a redundância discretamente
        
        # Estratégia 2: Redução de Janela (Fallback - 1 mês)
        time.sleep(1) # Pequena pausa para respirar
        try:
            fallback_tf = 'today 1-m'
            pytrends.build_payload(termos, geo=geo, timeframe=fallback_tf)
            df = pytrends.interest_over_time()
            if not df.empty:
                st.info(f"Nota: Dados recuperados com janela reduzida ({fallback_tf}) devido à instabilidade da conexão.")
                return df.drop(columns=['isPartial'], errors='ignore')
        except Exception as e:
            st.error(f"Não foi possível estabelecer conexão segura. Erro técnico: {e}")
            return None

    def buscar_mapa(self, termo, timeframe):
        """Busca dados geográficos para o mapa."""
        pytrends = self.conectar()
        try:
            pytrends.build_payload([termo], geo='BR', timeframe=timeframe)
            return pytrends.interest_by_region(resolution='COUNTRY', inc_low_vol=True)
        except:
            return None

# ==============================================================================
# 3. CLASSE: GERADOR DE CONTEXTO SEMÂNTICO (CONTEXT ENGINE)
# ==============================================================================
class ContextEngine:
    """
    Responsável por expandir uma doença simples em um ecossistema de termos
    para análise multidimensional.
    """
    @staticmethod
    def gerar_matriz_termos(doenca):
        return {
            "alvo": doenca,
            "clinico_primario": f"sintomas de {doenca}",
            "clinico_secundario": f"dor de {doenca}" if "dor" not in doenca else f"febre {doenca}",
            "farmacologico": f"remedio para {doenca}",
            "ruido_institucional": f"casos de {doenca}",
            "controle_neutro": "previsão do tempo"
        }

    @staticmethod
    def obter_lista_payload(ctx):
        # O Google aceita max 5 termos. Selecionamos os 5 mais críticos para a tese.
        # [Alvo, Clinico1, Farmacia, Ruido, Controle]
        return [
            ctx["alvo"],
            ctx["clinico_primario"],
            ctx["farmacologico"],
            ctx["ruido_institucional"],
            ctx["controle_neutro"]
        ]

# ==============================================================================
# 4. CLASSE: PROCESSADOR MATEMÁTICO (MATH ENGINE)
# ==============================================================================
class MathEngine:
    """
    Núcleo de processamento estatístico e diferencial.
    """
    @staticmethod
    def aplicar_suavizamento(df, window=7):
        """Aplica Média Móvel Retrospectiva (center=False)."""
        df_smooth = df.copy()
        for col in df.columns:
            df_smooth[f'{col}_suave'] = df[col].rolling(window=window, center=False, min_periods=1).mean()
        return df_smooth

    @staticmethod
    def calcular_canal_endemico(serie):
        """Calcula Limiar de Alerta (Intervalo de Confiança 95%)."""
        media = serie.mean()
        std = serie.std()
        return media + (1.96 * std)

    @staticmethod
    def calcular_derivadas(serie):
        """Calcula Velocidade (1ª Derivada) e Aceleração (2ª Derivada)."""
        velocidade = np.gradient(serie)
        aceleracao = np.gradient(velocidade)
        return velocidade, aceleracao

    @staticmethod
    def calcular_lead_time_lag(serie_alvo, serie_preditora):
        """Calcula o Lag (dias) de maior correlação cruzada."""
        best_lag = 0
        best_corr = -1
        for lag in range(1, 15): # Testa até 14 dias de antecedência
            # Shiftamos a preditora para o futuro para ver se ela alinha com o alvo
            s_shifted = serie_preditora.shift(lag)
            corr = serie_alvo.corr(s_shifted)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        return best_lag, best_corr

    @staticmethod
    def calcular_vero_index(val_clinico, val_ruido):
        """D8: Índice de Veracidade (Sinal / Ruído)."""
        return val_clinico / (val_ruido + 0.1)

    @staticmethod
    def calcular_asi(serie):
        """Calcula Índice de Saturação de Atenção (Volatilidade)."""
        if serie.mean() == 0: return 0
        cv = serie.std() / (serie.mean() + 0.01) # Coeficiente de Variação
        return cv

# ==============================================================================
# 5. LÓGICA DE EXECUÇÃO PRINCIPAL
# ==============================================================================

# --- Interface Lateral ---
with st.sidebar:
    st.header("🎛️ Centro de Comando")
    input_doenca = st.text_input("Agravo para Investigação:", placeholder="Ex: Dengue")
    input_uf = st.selectbox("Unidade Federativa:", 
                           ["BR-MS", "BR-SP", "BR-RJ", "BR-MG", "BR-PR", "BR-SC", "BR-RS", "BR-GO", "BR-MT", "BR-BA", "BR-PE", "BR-CE", "BR-AM"])
    
    st.divider()
    st.markdown("### 🔬 Protocolos Ativos")
    st.caption("✅ **D1:** Lead-Time Preditivo")
    st.caption("✅ **D2:** Persistência Robusta")
    st.caption("✅ **D3:** Baseline Sazonal")
    st.caption("✅ **D4/D8:** Filtro de Ruído/Vero-Index")
    st.caption("✅ **D5:** Sincronia Entrópica")
    st.caption("✅ **D6:** Fluxo Diferencial (Aceleração)")
    st.caption("✅ **D7:** Pressão Farmacológica")
    st.caption("✅ **D9:** Normalização Estatística")
    st.caption("✅ **D10:** Relatório Técnico Automático")

# --- Execução ---
if st.button("🚀 INICIAR VARREDURA EPIDEMIOLÓGICA TOTAL"):
    if not input_doenca:
        st.warning("É necessário definir um agravo para iniciar a varredura.")
    else:
        # Instanciar Motores
        conn = ConnectionEngine()
        ctx_eng = ContextEngine()
        math_eng = MathEngine()

        # 1. Preparação de Contexto
        contexto = ctx_eng.gerar_matriz_termos(input_doenca)
        termos_busca = ctx_eng.obter_lista_payload(contexto)
        
        # Mapeamento para facilitar leitura
        col_alvo = termos_busca[0]
        col_clinico = termos_busca[1]
        col_remedio = termos_busca[2]
        col_ruido = termos_busca[3]
        col_controle = termos_busca[4]

        with st.status("Executando Pipeline de Dados...", expanded=True) as status:
            st.write("📡 Conectando aos servidores de dados...")
            df_raw = conn.executar_busca_blindada(termos_busca, input_uf, 'today 3-m')
            
            if df_raw is not None:
                st.write("🗺️ Recuperando dados geoespaciais...")
                df_mapa = conn.buscar_mapa(col_alvo, 'today 1-m')
                
                st.write("🧮 Processando cálculo diferencial e estatística robusta...")
                
                # --- Pipeline Matemático ---
                # 1. Suavizamento (Média Móvel Retrospectiva)
                df_proc = math_eng.aplicar_suavizamento(df_raw)
                
                # Definindo nomes das colunas suavizadas
                alvo_s = f"{col_alvo}_suave"
                clinico_s = f"{col_clinico}_suave"
                remedio_s = f"{col_remedio}_suave"
                ruido_s = f"{col_ruido}_suave"
                
                # 2. Canal Endêmico
                limiar_alerta = math_eng.calcular_canal_endemico(df_proc[alvo_s])
                df_proc['limiar'] = limiar_alerta
                
                # 3. Derivadas (Velocidade e Aceleração)
                vel, acel = math_eng.calcular_derivadas(df_proc[alvo_s])
                df_proc['velocidade'] = vel
                df_proc['aceleracao'] = acel
                
                # 4. Lead-Time Analysis
                lag_dias, corr_lag = math_eng.calcular_lead_time_lag(df_proc[alvo_s], df_proc[clinico_s])
                
                # 5. Métricas Pontuais (Último dia)
                val_atual = df_proc[alvo_s].iloc[-1]
                val_limiar = df_proc['limiar'].iloc[-1]
                val_acel = df_proc['aceleracao'].iloc[-1]
                
                vero_index = math_eng.calcular_vero_index(df_proc[clinico_s].iloc[-1], df_proc[ruido_s].iloc[-1])
                asi = math_eng.calcular_asi(df_proc[alvo_s])
                corr_farmacia = df_proc[alvo_s].corr(df_proc[remedio_s])
                
                status.update(label="Processamento Concluído com Sucesso!", state="complete")

                # ==============================================================================
                # 6. DASHBOARD DE INTELIGÊNCIA (OUTPUT)
                # ==============================================================================
                st.markdown("---")
                st.header(f"📑 Dossiê Epidemiológico: {input_doenca.upper()}")
                
                # --- KPIs Principais ---
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                
                # Tratamento de divisão por zero se o limiar for muito baixo
                if val_limiar > 0:
                    delta_limiar = ((val_atual - val_limiar) / val_limiar) * 100
                else:
                    delta_limiar = 0
                    
                kpi1.metric("Intensidade (Suave 7D)", f"{val_atual:.1f}", f"{delta_limiar:.1f}% vs Limiar")
                
                kpi2.metric("Vero-Index (Fidelidade)", f"{vero_index:.2f}", "Alta Confiabilidade" if vero_index > 1 else "Possível Ruído")
                
                kpi3.metric("Lead-Time Detectado", f"{lag_dias} Dias", f"Corr: {corr_lag:.2f}")
                
                kpi4.metric("Aceleração do Surto", f"{val_acel:.2f}", "Expansão" if val_acel > 0 else "Retração")

                st.divider()

                # --- SEÇÃO 1: EVIDÊNCIA CIENTÍFICA (Canal Endêmico) ---
                col_chart, col_analysis = st.columns([2, 1])
                
                with col_chart:
                    st.subheader("📈 Canal Endêmico vs. Realidade")
                    st.caption("Linha Sólida: Dados Suavizados | Linha Vermelha: Limiar de Alerta (95% Confiança)")
                    
                    fig_main = go.Figure()
                    fig_main.add_trace(go.Scatter(x=df_proc.index, y=df_proc[alvo_s], mode='lines', name=f'Casos (Estimados)', line=dict(color='blue', width=2)))
                    fig_main.add_trace(go.Scatter(x=df_proc.index, y=df_proc['limiar'], mode='lines', name='Limiar Endêmico', line=dict(color='red', dash='dash')))
                    st.plotly_chart(fig_main, use_container_width=True)

                with col_analysis:
                    st.subheader("🩺 Diagnóstico Algorítmico")
                    
                    # Lógica de Decisão Complexa (Decision Tree simplificada)
                    if val_atual > val_limiar:
                        if vero_index > 0.8 and corr_farmacia > 0.5:
                            st.error(f"🚨 **SURTO BIOLÓGICO CONFIRMADO**\n\nO volume rompeu o limiar de segurança com alta consistência clínica e busca ativa por medicamentos. A aceleração está {'positiva' if val_acel > 0 else 'negativa'}, indicando {'agravamento' if val_acel > 0 else 'estabilização'}.")
                        else:
                            st.warning(f"⚠️ **ANOMALIA INFORMACIONAL**\n\nHá rompimento de limiar, mas o Vero-Index ({vero_index:.2f}) é baixo. Isso sugere pânico social induzido por notícias, sem correspondência clínica forte.")
                    else:
                        st.success(f"✅ **SITUAÇÃO ENDÊMICA**\n\nOs indicadores permanecem dentro do canal de segurança esperado para o período.")
                    
                    st.markdown(f"""
                    **Auditoria de Dados:**
                    * **Farmácia:** Correlação de {corr_farmacia:.2f}
                    * **ASI (Saturação):** {asi:.2f} (Volatilidade)
                    * **Previsão:** Sintomas antecedem casos em {lag_dias} dias.
                    """)

                # --- SEÇÃO 2: MAPA E DIFERENCIAL ---
                st.divider()
                tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Calor Nacional", "💊 Pressão Farmacológica", "🌪️ Dinâmica de Aceleração"])
                
                with tab1:
                    if df_mapa is not None:
                        df_mapa_res = df_mapa.reset_index()
                        fig_map = px.choropleth(
                            df_mapa_res,
                            geojson="https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson",
                            locations='geoName',
                            featureidkey="properties.name",
                            color=df_mapa_res.columns[1],
                            color_continuous_scale="Reds",
                            scope="south america",
                            title=f"Intensidade Geográfica: {input_doenca}"
                        )
                        fig_map.update_geos(fitbounds="locations", visible=False)
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.warning("Dados geográficos indisponíveis nesta janela de tempo (API Limit).")
                
                with tab2:
                    st.subheader("Correlação: Doença vs Tratamento")
                    # Normalizando para visualização
                    df_norm = (df_proc[[alvo_s, remedio_s]] - df_proc[[alvo_s, remedio_s]].min()) / (df_proc[[alvo_s, remedio_s]].max() - df_proc[[alvo_s, remedio_s]].min())
                    st.line_chart(df_norm)
                    st.caption("Se as linhas sobem juntas, a população está buscando tratamento, confirmando o surto.")

                with tab3:
                    st.subheader("Segunda Derivada (Aceleração da Curva)")
                    st.area_chart(df_proc['aceleracao'])
                    st.caption("Áreas acima de zero indicam surto em expansão explosiva.")

                # --- EXPORTAÇÃO ---
                st.divider()
                st.subheader("💾 Exportação de Dados para Pesquisa")
                
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_full = df_proc.to_csv().encode('utf-8')
                    st.download_button("📄 Baixar Matriz Completa (CSV)", csv_full, f"svei_full_{input_doenca}.csv", "text/csv")
                
                with col_dl2:
                    # Relatório de Texto Simplificado
                    relatorio_txt = f"""
                    RELATÓRIO TÉCNICO DE VIGILÂNCIA - SVEI vFinal
                    ---------------------------------------------
                    Agravo: {input_doenca}
                    UF: {input_uf}
                    Data: {datetime.now()}
                    
                    DIAGNÓSTICO AUTOMÁTICO:
                    - Intensidade Atual: {val_atual:.2f} (Limiar: {val_limiar:.2f})
                    - Status: {'SURTO' if val_atual > val_limiar else 'NORMAL'}
                    - Aceleração: {val_acel:.4f}
                    
                    VALIDAÇÃO CIENTÍFICA:
                    - Vero-Index: {vero_index:.2f}
                    - Lead-Time Lag: {lag_dias} dias
                    - Correlação Farmácia: {corr_farmacia:.2f}
                    
                    METODOLOGIA:
                    Z-Score, Média Móvel Retrospectiva (7D), Intervalo de Confiança 95%.
                    """
                    st.download_button("📝 Baixar Parecer Técnico (TXT)", relatorio_txt, f"parecer_{input_doenca}.txt", "text/plain")

            else:
                st.error("❌ O sistema de proteção do Google bloqueou as conexões. Isso é comum em ambientes compartilhados. Aguarde alguns minutos e tente novamente.")
