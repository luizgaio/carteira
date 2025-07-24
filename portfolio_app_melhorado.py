import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import time
from io import BytesIO
import base64

# Configuração da página
st.set_page_config(
    page_title="Análise de Portfólio Avançada", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/seu-usuario/portfolio-app',
        'Report a bug': "https://github.com/seu-usuario/portfolio-app/issues",
        'About': "# Análise de Portfólio\nFerramenta educacional para análise quantitativa de portfólios"
    }
)

# Suprimir warnings do yfinance
warnings.filterwarnings('ignore')

# CSS customizado para melhor aparência
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
    .portfolio-preset {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem;
        border-left: 4px solid #1f4e79;
    }
    .big-button {
        font-size: 1.2rem;
        padding: 0.7rem 1.5rem;
        margin: 1rem auto;
        display: block;
        width: 80%;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">📊 Análise de Portfólio Avançada</h1>', unsafe_allow_html=True)

# Dados dos ativos organizados por setor
@st.cache_data
def get_assets_by_sector():
    """Retorna ativos organizados por setor"""
    return {
        "Financeiro": {
            "BBAS3.SA": "Banco do Brasil",
            "ITUB4.SA": "Itaú Unibanco",
            "BBDC4.SA": "Bradesco",
            "SANB11.SA": "Santander",
            "BPAC11.SA": "BTG Pactual"
        },
        "Petróleo & Gás": {
            "PETR4.SA": "Petrobras PN",
            "PETR3.SA": "Petrobras ON",
            "PRIO3.SA": "PetroRio",
            "RRRP3.SA": "3R Petroleum"
        },
        "Mineração": {
            "VALE3.SA": "Vale",
            "CSNA3.SA": "CSN",
            "USIM5.SA": "Usiminas",
            "GGBR4.SA": "Gerdau"
        },
        "Varejo": {
            "MGLU3.SA": "Magazine Luiza",
            "LREN3.SA": "Lojas Renner",
            "ASAI3.SA": "Assaí",
            "PCAR3.SA": "P.Açúcar-CBD"
        },
        "Tecnologia": {
            "B3SA3.SA": "B3",
            "TOTS3.SA": "Totvs",
            "LWSA3.SA": "Locaweb",
            "POSI3.SA": "Positivo"
        },
        "Utilities": {
            "EGIE3.SA": "Engie Brasil",
            "CPFE3.SA": "CPFL Energia",
            "ELET3.SA": "Eletrobras",
            "SBSP3.SA": "Sabesp"
        },
        "Consumo": {
            "ABEV3.SA": "Ambev",
            "JBSS3.SA": "JBS",
            "BRFS3.SA": "BRF",
            "NATU3.SA": "Natura"
        }
    }

@st.cache_data
def get_preset_portfolios():
    """Retorna portfólios pré-definidos"""
    return {
        "Ibovespa Top 10": ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", 
                           "B3SA3.SA", "MGLU3.SA", "LREN3.SA", "JBSS3.SA", "BBAS3.SA"],
        "Financeiro": ["BBAS3.SA", "ITUB4.SA", "BBDC4.SA", "SANB11.SA", "BPAC11.SA"],
        "Dividendos": ["VALE3.SA", "ITUB4.SA", "BBDC4.SA", "EGIE3.SA", "CPFE3.SA"],
        "Small Caps": ["PRIO3.SA", "TOTS3.SA", "LWSA3.SA", "POSI3.SA", "RRRP3.SA"],
        "Conservador": ["ITUB4.SA", "BBDC4.SA", "EGIE3.SA", "SBSP3.SA", "CPFE3.SA"]
    }

def create_asset_selector():
    """Cria interface melhorada para seleção de ativos"""
    st.subheader("🎯 Seleção de Ativos")
    
    assets_by_sector = get_assets_by_sector()
    preset_portfolios = get_preset_portfolios()
    
    # Inicializa a sessão para armazenar os ativos selecionados
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = []
    
    # Abas para diferentes métodos de seleção
    tab1, tab2, tab3 = st.tabs(["🔍 Busca Manual", "📋 Portfólios Prontos", "🏢 Por Setor"])
    
    with tab1:
        # Busca manual com autocomplete
        all_assets = {}
        for sector_assets in assets_by_sector.values():
            all_assets.update(sector_assets)
        
        search_term = st.text_input("🔍 Buscar ativos (ticker ou nome da empresa)", 
                                   placeholder="Ex: VALE3, Petrobras, ITUB4...")
        
        if search_term:
            filtered_assets = {k: v for k, v in all_assets.items() 
                             if search_term.upper() in k.upper() or search_term.upper() in v.upper()}
            
            if filtered_assets:
                asset_options = [f"{ticker} - {name}" for ticker, name in filtered_assets.items()]
                selected_assets_display = st.multiselect(
                    "Ativos encontrados:", 
                    asset_options,
                    default=[f"{a} - {all_assets[a]}" for a in st.session_state.selected_assets if a in all_assets]
                )
                st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
            else:
                st.warning("Nenhum ativo encontrado para a busca.")
        else:
            # Seleção padrão
            default_assets = ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "B3SA3.SA"]
            asset_options = [f"{ticker} - {all_assets[ticker]}" for ticker in default_assets]
            selected_assets_display = st.multiselect(
                "Selecione os ativos:", 
                [f"{ticker} - {name}" for ticker, name in all_assets.items()],
                default=[f"{a} - {all_assets[a]}" for a in st.session_state.selected_assets if a in all_assets]
            )
            st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
    
    with tab2:
        # Portfólios pré-definidos
        st.markdown("Selecione um portfólio pré-configurado:")
        
        cols = st.columns(3)
        
        for i, (name, assets) in enumerate(preset_portfolios.items()):
            with cols[i % 3]:
                if st.button(f"📊 {name}", key=f"preset_{i}"):
                    st.session_state.selected_assets = assets
                    st.success(f"Portfólio selecionado: {len(assets)} ativos")
        
        # Mostrar ativos selecionados se houver
        if st.session_state.selected_assets:
            st.markdown("**Ativos selecionados:**")
            for asset in st.session_state.selected_assets:
                asset_name = all_assets.get(asset, asset)
                st.markdown(f"<div class='portfolio-preset'>• {asset} - {asset_name}</div>", 
                           unsafe_allow_html=True)
    
    with tab3:
        # Seleção por setor
        selected_sector = st.selectbox("Escolha o setor:", list(assets_by_sector.keys()))
        
        if selected_sector:
            sector_assets = assets_by_sector[selected_sector]
            asset_options = [f"{ticker} - {name}" for ticker, name in sector_assets.items()]
            selected_assets_display = st.multiselect(
                f"Ativos do setor {selected_sector}:", 
                asset_options,
                default=[f"{a} - {sector_assets[a]}" for a in st.session_state.selected_assets if a in sector_assets]
            )
            st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
    
    return st.session_state.selected_assets

@st.cache_data(ttl=3600, show_spinner="📥 Baixando dados do mercado...")
def download_data_optimized(tickers, start_date, end_date):
    """Download otimizado de dados com cache e tratamento de erros"""
    try:
        if not tickers:
            return None, None
        
        # Adicionar benchmark
        all_tickers = tickers + ['^BVSP']
        
        # Download em batch
        data = yf.download(all_tickers, start=start_date, end=end_date, 
                          auto_adjust=True, progress=False)
        
        if data.empty:
            st.error("❌ Nenhum dado encontrado para o período selecionado")
            return None, None
        
        # Extrair preços de fechamento - CORREÇÃO PRINCIPAL AQUI
        if len(all_tickers) == 1:
            # Caso especial para um único ativo
            prices = pd.DataFrame(data['Close']).rename(columns={'Close': all_tickers[0]})
        else:
            prices = data['Close']
        
        # Verificar dados faltantes
        missing_pct = prices.isnull().sum() / len(prices) * 100
        problematic_assets = missing_pct[missing_pct > 10]
        
        if not problematic_assets.empty:
            st.warning(f"⚠️ Ativos com muitos dados faltantes (>10%): {', '.join(problematic_assets.index)}")
        
        # Remover linhas com muitos NaN
        prices = prices.dropna()
        
        if len(prices) < 50:
            st.error("❌ Dados insuficientes para análise (mínimo 50 observações)")
            return None, None
        
        return prices, None
        
    except Exception as e:
        st.error(f"❌ Erro ao baixar dados: {str(e)}")
        return None, None

@st.cache_data(ttl=1800)
def calculate_returns_and_metrics(prices, tickers):
    """Calcula retornos e métricas básicas com cache"""
    try:
        # Calcular retornos logarítmicos
        returns = np.log(prices[tickers] / prices[tickers].shift(1)).dropna()
        benchmark_returns = np.log(prices['^BVSP'] / prices['^BVSP'].shift(1)).dropna()
        
        # Métricas anualizadas
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        return returns, benchmark_returns, mean_returns, cov_matrix
        
    except Exception as e:
        st.error(f"❌ Erro no cálculo de retornos: {str(e)}")
        return None, None, None, None

@st.cache_data(ttl=1800)
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations=10000):
    """Simulação Monte Carlo otimizada com cache"""
    try:
        num_assets = len(mean_returns)
        
        # Gerar pesos aleatórios usando Dirichlet (mais eficiente)
        weights = np.random.dirichlet(np.ones(num_assets), num_simulations)
        
        # Calcular retornos e riscos vetorizados
        portfolio_returns = np.dot(weights, mean_returns.values)
        portfolio_risks = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix.values, weights))
        
        return weights, portfolio_returns, portfolio_risks
        
    except Exception as e:
        st.error(f"❌ Erro na simulação Monte Carlo: {str(e)}")
        return None, None, None

def calculate_portfolio_metrics(returns, benchmark_returns, weights, risk_free_rate):
    """Calcula métricas detalhadas do portfólio"""
    try:
        portfolio_returns = returns @ weights
        
        # Métricas básicas
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        # Sortino Ratio (usando apenas downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = np.inf
        
        # Beta
        aligned_benchmark = benchmark_returns.loc[portfolio_returns.index]
        if len(aligned_benchmark) > 0:
            covariance = np.cov(portfolio_returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Treynor Ratio
            treynor_ratio = (annual_return - risk_free_rate) / beta if beta != 0 else 0
        else:
            beta = 0
            treynor_ratio = 0
        
        # VaR e CVaR
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_95 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'treynor_ratio': treynor_ratio,
            'beta': beta,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'portfolio_returns': portfolio_returns
        }
        
    except Exception as e:
        st.error(f"❌ Erro no cálculo de métricas: {str(e)}")
        return None

def create_advanced_charts(prices, returns, benchmark_returns, weights, portfolio_metrics, 
                          mc_weights, mc_returns, mc_risks, risk_free_rate):
    """Cria visualizações avançadas"""
    
    charts = {}
    
    try:
        # 1. Preços normalizados (base 100)
        normalized_prices = (prices / prices.iloc[0]) * 100
        fig_prices = px.line(normalized_prices, title="📈 Evolução dos Preços (Base 100)")
        fig_prices.update_layout(xaxis_title="Data", yaxis_title="Preço Normalizado")
        charts['prices'] = fig_prices
        
        # 2. Fronteira Eficiente Avançada
        fig_frontier = go.Figure()
        
        # Pontos da simulação Monte Carlo
        sharpe_ratios = (mc_returns - risk_free_rate) / mc_risks
        fig_frontier.add_trace(go.Scatter(
            x=mc_risks,
            y=mc_returns,
            mode='markers',
            marker=dict(
                color=sharpe_ratios,
                colorscale='Viridis',
                size=4,
                opacity=0.6,
                colorbar=dict(title="Sharpe Ratio")
            ),
            name='Carteiras Simuladas',
            hovertemplate='Risco: %{x:.2%}<br>Retorno: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>'
        ))
        
        # Carteira selecionada
        fig_frontier.add_trace(go.Scatter(
            x=[portfolio_metrics['annual_volatility']],
            y=[portfolio_metrics['annual_return']],
            mode='markers+text',
            marker=dict(color='red', size=15, symbol='star'),
            text=["Carteira Selecionada"],
            textposition="top center",
            name="Carteira Selecionada"
        ))
        
        # Benchmark (Ibovespa)
        ibov_return = benchmark_returns.mean() * 252
        ibov_risk = benchmark_returns.std() * np.sqrt(252)
        fig_frontier.add_trace(go.Scatter(
            x=[ibov_risk],
            y=[ibov_return],
            mode='markers+text',
            marker=dict(color='blue', size=12, symbol='diamond'),
            text=["Ibovespa"],
            textposition="top center",
            name="Ibovespa"
        ))
        
        fig_frontier.update_layout(
            title="🎯 Fronteira Eficiente com Análise de Sharpe",
            xaxis_title="Volatilidade (Risco)",
            yaxis_title="Retorno Esperado",
            hovermode='closest'
        )
        charts['frontier'] = fig_frontier
        
        # 3. Desempenho Acumulado Comparativo
        portfolio_cumulative = (1 + portfolio_metrics['portfolio_returns']).cumprod() * 100
        benchmark_aligned = benchmark_returns.loc[portfolio_metrics['portfolio_returns'].index]
        benchmark_cumulative = (1 + benchmark_aligned).cumprod() * 100
        
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(
            x=portfolio_cumulative.index,
            y=portfolio_cumulative,
            name="Carteira",
            line=dict(color='green', width=2)
        ))
        fig_performance.add_trace(go.Scatter(
            x=benchmark_cumulative.index,
            y=benchmark_cumulative,
            name="Ibovespa",
            line=dict(color='blue', width=2)
        ))
        
        fig_performance.update_layout(
            title="📊 Desempenho Acumulado (Base 100)",
            xaxis_title="Data",
            yaxis_title="Valor Acumulado"
        )
        charts['performance'] = fig_performance
        
        # 4. Análise de Drawdown
        cumulative_returns = (1 + portfolio_metrics['portfolio_returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig_drawdown = go.Figure()
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tonexty',
            name="Drawdown da Carteira",
            line=dict(color='red')
        ))
        
        fig_drawdown.update_layout(
            title="📉 Análise de Drawdown",
            xaxis_title="Data",
            yaxis_title="Drawdown (%)",
            yaxis=dict(tickformat='.1%')
        )
        charts['drawdown'] = fig_drawdown
        
        # 5. Volatilidade Móvel (30 dias)
        portfolio_vol = portfolio_metrics['portfolio_returns'].rolling(30).std() * np.sqrt(252)
        benchmark_vol = benchmark_aligned.rolling(30).std() * np.sqrt(252)
        
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(
            x=portfolio_vol.index,
            y=portfolio_vol,
            name="Carteira",
            line=dict(color='orange')
        ))
        fig_volatility.add_trace(go.Scatter(
            x=benchmark_vol.index,
            y=benchmark_vol,
            name="Ibovespa",
            line=dict(color='blue')
        ))
        
        fig_volatility.update_layout(
            title="📊 Volatilidade Móvel (30 dias)",
            xaxis_title="Data",
            yaxis_title="Volatilidade Anualizada"
        )
        charts['volatility'] = fig_volatility
        
        # 6. Distribuição de Retornos
        fig_distribution = go.Figure()
        fig_distribution.add_trace(go.Histogram(
            x=portfolio_metrics['portfolio_returns'],
            nbinsx=50,
            name="Retornos da Carteira",
            opacity=0.7
        ))
        
        # Adicionar linhas de VaR
        fig_distribution.add_vline(
            x=np.percentile(portfolio_metrics['portfolio_returns'], 5),
            line_dash="dash",
            line_color="red",
            annotation_text="VaR 95%"
        )
        
        fig_distribution.update_layout(
            title="📊 Distribuição de Retornos Diários",
            xaxis_title="Retorno Diário",
            yaxis_title="Frequência"
        )
        charts['distribution'] = fig_distribution
        
        # 7. Composição da Carteira
        asset_names = [ticker.replace('.SA', '') for ticker in returns.columns]
        fig_composition = px.pie(
            values=weights,
            names=asset_names,
            title="🥧 Composição da Carteira"
        )
        charts['composition'] = fig_composition
        
        # 8. Matriz de Correlação
        correlation_matrix = returns.corr()
        fig_correlation = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale='RdBu_r',
            title="🔗 Matriz de Correlação dos Ativos"
        )
        charts['correlation'] = fig_correlation
        
        return charts
        
    except Exception as e:
        st.error(f"❌ Erro na criação de gráficos: {str(e)}")
        return {}

def export_results(charts, portfolio_metrics, weights, asset_names):
    """Funcionalidade de exportação melhorada"""
    
    st.subheader("📥 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Formato de exportação:",
            ["PNG (Gráficos)", "CSV (Dados)", "Relatório Resumo"]
        )
    
    with col2:
        if st.button("📥 Exportar", type="primary"):
            if export_format == "PNG (Gráficos)":
                # Exportar gráficos como PNG
                for chart_name, chart in charts.items():
                    img_bytes = chart.to_image(format="png", width=1200, height=800, scale=2)
                    st.download_button(
                        f"Download {chart_name.title()}",
                        img_bytes,
                        f"portfolio_{chart_name}.png",
                        "image/png",
                        key=f"download_{chart_name}"
                    )
            
            elif export_format == "CSV (Dados)":
                # Criar DataFrame com resultados
                results_df = pd.DataFrame({
                    'Ativo': asset_names,
                    'Peso': weights,
                    'Peso_Percentual': [f"{w:.2%}" for w in weights]
                })
                
                csv_buffer = BytesIO()
                results_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    "Download Dados CSV",
                    csv_buffer.getvalue(),
                    "portfolio_data.csv",
                    "text/csv"
                )
            
            elif export_format == "Relatório Resumo":
                # Criar relatório em texto
                report = f"""
RELATÓRIO DE ANÁLISE DE PORTFÓLIO
================================

Data da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}

COMPOSIÇÃO DA CARTEIRA:
{'-' * 30}
"""
                for i, asset in enumerate(asset_names):
                    report += f"{asset}: {weights[i]:.2%}\n"
                
                report += f"""

MÉTRICAS DE PERFORMANCE:
{'-' * 30}
Retorno Anualizado: {portfolio_metrics['annual_return']:.2%}
Volatilidade Anualizada: {portfolio_metrics['annual_volatility']:.2%}
Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}
Sortino Ratio: {portfolio_metrics['sortino_ratio']:.2f}
Beta: {portfolio_metrics['beta']:.2f}
VaR (95%): {portfolio_metrics['var_95']:.2%}
CVaR (95%): {portfolio_metrics['cvar_95']:.2%}
Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}
Calmar Ratio: {portfolio_metrics['calmar_ratio']:.2f}
"""
                
                st.download_button(
                    "Download Relatório",
                    report,
                    "portfolio_report.txt",
                    "text/plain"
                )

def main():
    """Função principal do aplicativo"""
    # Inicializa variáveis de sessão
    if 'analyze' not in st.session_state:
        st.session_state.analyze = False
    
    # Sidebar com parâmetros
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Parâmetros de data
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "📅 Data de início",
                value=datetime.today() - timedelta(days=3*365),
                max_value=datetime.today() - timedelta(days=30)
            )
        
        with col2:
            end_date = st.date_input(
                "📅 Data de fim",
                value=datetime.today(),
                max_value=datetime.today()
            )
        
        # Taxa livre de risco
        risk_free_rate = st.number_input(
            "💰 Taxa Livre de Risco (% a.a.)",
            min_value=0.0,
            max_value=50.0,
            value=10.75,
            step=0.25,
            help="Taxa Selic atual ou taxa livre de risco desejada"
        ) / 100
        
        # Tipo de carteira
        portfolio_type = st.selectbox(
            "📊 Estratégia de Otimização",
            ["Máximo Sharpe", "Máximo Sortino", "Máximo Treynor", "Mínima Volatilidade", "Carteira Própria"],
            help="Critério para otimização da carteira"
        )
        
        # Número de simulações
        num_simulations = st.slider(
            "🎲 Simulações Monte Carlo",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Mais simulações = maior precisão, mas processamento mais lento"
        )
    
    # Validação de inputs
    if start_date >= end_date:
        st.error("❌ A data de início deve ser anterior à data de fim")
        return
    
    if (end_date - start_date).days < 252:
        st.warning("⚠️ Período muito curto pode resultar em análise pouco confiável (recomendado: mínimo 1 ano)")
    
    # Seleção de ativos
    selected_assets = create_asset_selector()

    if not selected_assets or len(selected_assets) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos para análise")
        st.stop()

    # Exibe botão e espera o clique
    if not st.session_state.analyze:
        if st.button("🚀 Analisar Portfólio", type="primary", use_container_width=True):
            st.session_state.analyze = True
        else:
            st.info("💡 Selecione os ativos e clique em 'Analisar Portfólio' para continuar")
            st.stop()
    
    if len(selected_assets) > 15:
        st.warning("⚠️ Muitos ativos podem impactar a performance. Considere reduzir para menos de 15.")
    
    # Download e processamento de dados
    with st.spinner("🔄 Processando dados..."):
        prices, error = download_data_optimized(selected_assets, start_date, end_date)
        
        if prices is None:
            return
        
        returns, benchmark_returns, mean_returns, cov_matrix = calculate_returns_and_metrics(prices, selected_assets)
        
        if returns is None:
            return
    
    # Simulação Monte Carlo
    with st.spinner("🎲 Executando simulação Monte Carlo..."):
        mc_weights, mc_returns, mc_risks = monte_carlo_simulation(mean_returns, cov_matrix, num_simulations)
        
        if mc_weights is None:
            return
    
    # Seleção da carteira ótima
    if portfolio_type == "Carteira Própria":
        st.subheader("⚖️ Defina os Pesos da Carteira")
        st.info("💡 A soma dos pesos deve ser igual a 100%")
        
        weights = []
        cols = st.columns(min(len(selected_assets), 3))
        
        for i, asset in enumerate(selected_assets):
            with cols[i % 3]:
                weight = st.number_input(
                    f"{asset.replace('.SA', '')}",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(selected_assets),
                    step=0.01,
                    format="%.2f",
                    key=f"weight_{asset}"
                )
                weights.append(weight)
        
        weights = np.array(weights)
        
        if not np.isclose(weights.sum(), 1.0, atol=0.01):
            st.error(f"❌ A soma dos pesos deve ser 1.0 (atual: {weights.sum():.2f})")
            return
    
    else:
        # Otimização automática
        if portfolio_type == "Máximo Sharpe":
            sharpe_ratios = (mc_returns - risk_free_rate) / mc_risks
            optimal_idx = np.argmax(sharpe_ratios)
        
        elif portfolio_type == "Máximo Sortino":
            # Calcular Sortino para cada carteira simulada
            sortino_ratios = []
            for i in range(len(mc_weights)):
                portfolio_returns = returns @ mc_weights[i]
                downside_returns = portfolio_returns[portfolio_returns < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std() * np.sqrt(252)
                    sortino = (mc_returns[i] - risk_free_rate) / downside_deviation
                else:
                    sortino = np.inf
                sortino_ratios.append(sortino)
            
            optimal_idx = np.argmax(sortino_ratios)
        
        elif portfolio_type == "Máximo Treynor":
            treynor_ratios = []
            for i in range(len(mc_weights)):
                portfolio_returns = returns @ mc_weights[i]
                aligned_benchmark = benchmark_returns.loc[portfolio_returns.index]
                if len(aligned_benchmark) > 0:
                    covariance = np.cov(portfolio_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                    treynor = (mc_returns[i] - risk_free_rate) / beta if beta != 0 else 0
                else:
                    treynor = 0
                treynor_ratios.append(treynor)
            
            optimal_idx = np.argmax(treynor_ratios)
        
        elif portfolio_type == "Mínima Volatilidade":
            optimal_idx = np.argmin(mc_risks)
        
        weights = mc_weights[optimal_idx]
    
    # Calcular métricas da carteira
    portfolio_metrics = calculate_portfolio_metrics(returns, benchmark_returns, weights, risk_free_rate)
    
    if portfolio_metrics is None:
        return
    
    # Exibir métricas principais
    st.subheader("📊 Indicadores da Carteira")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Retorno Anualizado", f"{portfolio_metrics['annual_return']:.2%}")
        st.metric("📊 Volatilidade", f"{portfolio_metrics['annual_volatility']:.2%}")
        st.metric("⚡ Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
    
    with col2:
        st.metric("📉 Sortino Ratio", f"{portfolio_metrics['sortino_ratio']:.2f}")
        st.metric("🎯 Treynor Ratio", f"{portfolio_metrics['treynor_ratio']:.2f}")
        st.metric("📊 Beta", f"{portfolio_metrics['beta']:.2f}")
    
    with col3:
        st.metric("⚠️ VaR (95%)", f"{portfolio_metrics['var_95']:.2%}")
        st.metric("🔻 CVaR (95%)", f"{portfolio_metrics['cvar_95']:.2%}")
        st.metric("📉 Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2%}")
    
    with col4:
        st.metric("🏆 Calmar Ratio", f"{portfolio_metrics['calmar_ratio']:.2f}")
        
        # Classificação de risco
        if portfolio_metrics['annual_volatility'] < 0.15:
            risk_level = "🟢 Baixo"
        elif portfolio_metrics['annual_volatility'] < 0.25:
            risk_level = "🟡 Moderado"
        else:
            risk_level = "🔴 Alto"
        
        st.metric("🎚️ Nível de Risco", risk_level)
    
    # Tabela de composição
    st.subheader("📋 Composição da Carteira")
    
    composition_df = pd.DataFrame({
        'Ativo': [asset.replace('.SA', '') for asset in selected_assets],
        'Ticker Completo': selected_assets,
        'Peso': weights,
        'Peso %': [f"{w:.2%}" for w in weights]
    })
    
    st.dataframe(composition_df, use_container_width=True)
    
    # Criar e exibir gráficos
    st.subheader("📊 Análises Visuais")
    
    charts = create_advanced_charts(
        prices, returns, benchmark_returns, weights, portfolio_metrics,
        mc_weights, mc_returns, mc_risks, risk_free_rate
    )
    
    if charts:
        # Organizar gráficos em abas
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🎯 Otimização", "📊 Risco", "🔗 Correlação"])
        
        with tab1:
            st.plotly_chart(charts['prices'], use_container_width=True)
            st.plotly_chart(charts['performance'], use_container_width=True)
        
        with tab2:
            st.plotly_chart(charts['frontier'], use_container_width=True)
            st.plotly_chart(charts['composition'], use_container_width=True)
        
        with tab3:
            st.plotly_chart(charts['drawdown'], use_container_width=True)
            st.plotly_chart(charts['volatility'], use_container_width=True)
            st.plotly_chart(charts['distribution'], use_container_width=True)
        
        with tab4:
            st.plotly_chart(charts['correlation'], use_container_width=True)
    
    # Funcionalidade de exportação
    export_results(charts, portfolio_metrics, weights, [asset.replace('.SA', '') for asset in selected_assets])
    
    # Insights e recomendações
    st.subheader("💡 Insights e Recomendações")
    
    insights = []
    
    # Análise de Sharpe
    if portfolio_metrics['sharpe_ratio'] > 1.0:
        insights.append("🟢 **Excelente relação risco-retorno** - Sharpe Ratio acima de 1.0 indica boa eficiência.")
    elif portfolio_metrics['sharpe_ratio'] > 0.5:
        insights.append("🟡 **Relação risco-retorno moderada** - Considere otimizações para melhorar o Sharpe Ratio.")
    else:
        insights.append("🔴 **Relação risco-retorno baixa** - Carteira pode não estar compensando adequadamente o risco.")
    
    # Análise de concentração
    max_weight = np.max(weights)
    if max_weight > 0.4:
        insights.append(f"⚠️ **Alta concentração** - {max_weight:.1%} em um único ativo pode aumentar o risco específico.")
    
    # Análise de drawdown
    if abs(portfolio_metrics['max_drawdown']) > 0.3:
        insights.append("🔴 **Alto drawdown máximo** - Carteira experimentou perdas significativas em algum período.")
    elif abs(portfolio_metrics['max_drawdown']) < 0.15:
        insights.append("🟢 **Drawdown controlado** - Carteira demonstrou boa resistência a perdas.")
    
    # Análise de beta
    if portfolio_metrics['beta'] > 1.2:
        insights.append("📈 **Alta sensibilidade ao mercado** - Carteira tende a amplificar movimentos do Ibovespa.")
    elif portfolio_metrics['beta'] < 0.8:
        insights.append("📉 **Baixa sensibilidade ao mercado** - Carteira tende a ser mais defensiva que o Ibovespa.")
    
    for insight in insights:
        st.markdown(insight)
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Análise de Portfólio Avançada</strong></p>
        <p>Desenvolvido pelo Prof. Luiz Eduardo Gaio para fins educacionais | Versão 2.0</p>
        <p><em>⚠️ Esta ferramenta é apenas para fins educacionais. Não constitui recomendação de investimento.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


