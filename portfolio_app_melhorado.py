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
    
    # Inicializa selected_assets no session_state se não existir
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
                selected_assets_display = st.multiselect("Ativos encontrados:", asset_options)
                st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
            else:
                st.warning("Nenhum ativo encontrado para a busca.")
                st.session_state.selected_assets = []
        else:
            # Seleção padrão
            default_assets = ["VALE3.SA", "PETR4.SA", "ITUB4.SA", "B3SA3.SA"]
            asset_options = [f"{ticker} - {all_assets[ticker]}" for ticker in default_assets]
            selected_assets_display = st.multiselect("Selecione os ativos:", 
                                                    [f"{ticker} - {name}" for ticker, name in all_assets.items()],
                                                    default=asset_options)
            st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
    
    with tab2:
        # Portfólios pré-definidos
        st.markdown("Selecione um portfólio pré-configurado:")
        cols = st.columns(3)
        selected_preset = None
        
        for i, (name, assets) in enumerate(preset_portfolios.items()):
            with cols[i % 3]:
                if st.button(f"📊 {name}", key=f"preset_{i}"):
                    selected_preset = assets
        
        if selected_preset:
            st.session_state.selected_assets = selected_preset
            st.success(f"Portfólio selecionado: {len(st.session_state.selected_assets)} ativos")
            
            # Mostrar ativos selecionados
            for asset in st.session_state.selected_assets:
                asset_name = all_assets.get(asset, asset)
                st.markdown(f"<div class=\'portfolio-preset\'>• {asset} - {asset_name}</div>", 
                           unsafe_allow_html=True)
        else:
            # Se nenhum preset foi selecionado, garantir que a lista de ativos não seja limpa
            # Mantém o estado anterior ou define como vazio se for a primeira execução
            if 'selected_assets' not in st.session_state:
                st.session_state.selected_assets = []

    with tab3:
        # Seleção por setor
        st.markdown("Selecione ativos por setor:")
        selected_sector = st.selectbox("Setor:", list(assets_by_sector.keys()))
        
        if selected_sector:
            sector_assets = assets_by_sector[selected_sector]
            asset_options = [f"{ticker} - {name}" for ticker, name in sector_assets.items()]
            selected_assets_display = st.multiselect(f"Ativos do setor {selected_sector}:", asset_options)
            st.session_state.selected_assets = [asset.split(" - ")[0] for asset in selected_assets_display]
        else:
            st.session_state.selected_assets = []
    
    # A função não retorna mais selected_assets, pois agora está no session_state
    # return selected_assets

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
        
        # Extrair preços de fechamento
        if len(all_tickers) == 1:
            prices = pd.DataFrame({all_tickers[0]: data["Close"]})
        else:
            prices = data["Close"]
        
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

