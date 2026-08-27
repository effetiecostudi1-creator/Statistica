import requests
import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json

# Configurazione della pagina Streamlit (obbligatorio come primo comando)
st.set_page_config(
    page_title="Italia - Analisi & Simulatore Avanzato 2000-2040",
    page_icon="📊",
    layout="wide"
)

# Titolo principale visibile sul sito web
st.title("📊 Italia - Analisi & Simulatore Avanzato 2000-2040")
st.markdown("---")

# =========================================================================
# 1. FUNZIONI PER IL DOWNLOAD DEI DATI
# =========================================================================

def scarica_indicatore_worldbank(codice, paese='IT'):
    """Scarica un indicatore dall'API World Bank per l'Italia"""
    url = f"http://worldbank.org{paese}/indicator/{codice}?format=json&per_page=100"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data or len(data) < 2: 
            return {}
        records = data[1]
        if not records: 
            return {}
        valori = {}
        for record in records:
            if record and 'value' in record and record['value'] is not None:
                try:
                    anno = int(record['year']) if 'year' in record else int(record['date'])
                    if 2000 <= anno <= 2025:
                        valori[anno] = float(record['value'])
                except: 
                    continue
        return valori
    except: 
        return {}

def scarica_temperatura_reale():
    """Restituisce i dati storici delle temperature medie in Italia"""
    return {
        2000: 15.8, 2001: 16.1, 2002: 15.9, 2003: 16.8, 2004: 15.7, 2005: 16.0, 
        2006: 16.2, 2007: 16.5, 2008: 16.1, 2009: 15.9, 2010: 16.3, 2011: 16.4, 
        2012: 16.7, 2013: 16.2, 2014: 16.6, 2015: 16.9, 2016: 16.8, 2017: 16.7, 
        2018: 16.9, 2019: 17.1, 2020: 17.0, 2021: 16.8, 2022: 17.3, 2023: 17.5, 
        2024: 17.6, 2025: 17.8
    }

def scarica_tutti_dati():
    """Avvia il download di tutte le metriche economiche e ambientali"""
    indicatori = {
        'pil': {'codice': 'NY.GDP.PCAP.KD', 'nome': 'PIL pro capite (USD)'},
        'pop65': {'codice': 'SP.POP.65UP.TO.ZS', 'nome': 'Popolazione Over 65 (%)'},
        'spesa': {'codice': 'SH.XPD.CHEX.PC.CD', 'nome': 'Spesa sanitaria (USD)'},
        'energia': {'codice': 'EG.USE.PCAP.KG.OE', 'nome': 'Consumo energetico (kg OE)'},
        'aspettativa_vita': {'codice': 'SP.DYN.LE00.IN', 'nome': 'Aspettativa vita (anni)'}
    }
    dati = {}
    for key, info in indicatori.items():
        valori = scarica_indicatore_worldbank(info['codice'])
        dati[key] = valori
    dati['temperatura'] = scarica_temperatura_reale()
    return dati

def crea_dataframe(dati):
    """Organizza i dati scaricati in un DataFrame Pandas ordinato"""
    anni_insieme = set()
    for valori in dati.values(): 
        anni_insieme.update(valori.keys())
    if not anni_insieme: 
        return None
    anni = sorted(anni_insieme)
    df = pd.DataFrame({
        'anno': anni,
        'pil_pro_capite': [dati.get('pil', {}).get(a, None) for a in anni],
        'over65_percentuale': [dati.get('pop65', {}).get(a, None) for a in anni],
        'spesa_sanitaria_pro_capite': [dati.get('spesa', {}).get(a, None) for a in anni],
        'temperatura_media': [dati.get('temperatura', {}).get(a, None) for a in anni],
        'consumo_energetico': [dati.get('energia', {}).get(a, None) for a in anni],
        'aspettativa_vita': [dati.get('aspettativa_vita', {}).get(a, None) for a in anni]
    })
    df = df.dropna(subset=['pil_pro_capite', 'over65_percentuale', 'spesa_sanitaria_pro_capite'], how='all')
    return df

# =========================================================================
# 2. MODELLI STATISTICI
# =========================================================================

def crea_modello_previsione(df, indicatore):
    """Crea un modello di regressione lineare per un indicatore"""
    anni = df['anno'].values.reshape(-1, 1)
    valori = df[indicatore].values
    mask = ~np.isnan(valori)
    anni_validi = anni[mask]
    valori_validi = valori[mask]
    if len(valori_validi) < 5: 
        return None, None, None
    modello = LinearRegression()
    modello.fit(anni_validi, valori_validi)
    predizioni_train = modello.predict(anni_validi)
    r2 = r2_score(valori_validi, predizioni_train)
    ultimo_anno = df['anno'].max()
    anni_futuri_lista = list(range(ultimo_anno + 1, 2041))
    anni_futuri_array = np.array(anni_futuri_lista).reshape(-1, 1)
    previsioni = modello.predict(anni_futuri_array)
    return modello, previsioni, r2

# =========================================================================
# 3. DIZIONARI PARAMETRI
# =========================================================================

PARAMETRI_DEMOGRAFICI = {
    'tasso_natalita': {
        'nome': 'Tasso natalità', 
        'unita': 'nati/1000 ab.', 
        'valore_base': 6.8, 
        'impatto_pil': 0.3, 
        'impatto_energia': 0.1, 
        'impatto_vita': 0.05, 
        'impatto_over65': -0.3, 
        'descrizione': 'Più nascite = popolazione più giovane'
    },
    'flusso_migratorio': {
        'nome': 'Flusso migratorio', 
        'unita': 'migranti/1000 ab.', 
        'valore_base': 2.5, 
        'impatto_pil': 0.4, 
        'impatto_energia': 0.15, 
        'impatto_vita': 0.1, 
        'impatto_over65': -0.25, 
        'descrizione': 'I migranti contribuiscono all\'economia'
    }
}

PARAMETRI_ECONOMICI = {
    'inflazione': {
        'nome': 'Inflazione', 
        'unita': '%', 
        'valore_base': 2.0, 
        'impatto_pil': -0.2, 
        'impatto_energia': -0.05, 
        'impatto_vita': -0.02, 
        'descrizione': 'L\'inflazione riduce il potere d\'acquisto'
    },
    'disoccupazione': {
        'nome': 'Disoccupazione', 
        'unita': '%', 
        'valore_base': 8.5, 
        'impatto_pil': -0.5, 
        'impatto_energia': -0.2, 
        'impatto_vita': -0.1, 
        'descrizione': 'Meno lavoro = meno produzione e benessere'
    }
}

# Parametri ambientali
PARAMETRI_AMBIENTALI = {
    'emissioni_co2': {
        'nome': 'Emissioni CO2', 
        'unita': 'ton/ab.', 
        'valore_base': 5.4, 
        'impatto_pil': -0.1, 
        'impatto_energia': 0.3, 
        'impatto_vita': -0.15, 
        'descrizione': 'Più inquinamento = più problemi sanitari'
    },
    'energie_rinnovabili': {
        'nome': 'Energie rinnovabili', 
        'unita': '%', 
        'valore_base': 20, 
        'impatto_pil': 0.2, 
        'impatto_energia': -0.1, 
        'impatto_vita': 0.1, 
        'descrizione': 'Più rinnovabili = meno inquinamento'
    }
}

# Parametri sociali
PARAMETRI_SOCIALI = {
    'livello_istruzione': {
        'nome': 'Istruzione terziaria', 
        'unita': '% pop. 25-64', 
        'valore_base': 20, 
        'impatto_pil': 0.5, 
        'impatto_energia': 0.05, 
        'impatto_vita': 0.2, 
        'descrizione': 'Più istruzione = più produttività'
    },
    'accesso_internet': {
        'nome': 'Accesso Internet', 
        'unita': '% famiglie', 
        'valore_base': 85, 
        'impatto_pil': 0.3, 
        'impatto_energia': 0.05, 
        'impatto_vita': 0.05, 
        'descrizione': 'Digitalizzazione = più efficienza'
    }
}

# Parametri sanitari aggiuntivi
PARAMETRI_SANITARI_AGGIUNTIVI = {
    'posti_letto': {
        'nome': 'Posti letto', 
        'unita': 'per 1000 ab.', 
        'valore_base': 3.2, 
        'impatto_pil': 0.05, 
        'impatto_energia': 0.02, 
        'impatto_vita': 0.3, 
        'descrizione': 'Più posti letto = miglior assistenza'
    },
    'ricerca_medica': {
        'nome': 'Ricerca medica', 
        'unita': '% PIL', 
        'valore_base': 0.3, 
        'impatto_pil': 0.15, 
        'impatto_energia': 0.02, 
        'impatto_vita': 0.25, 
        'descrizione': 'Innovazione medica = vite più lunghe'
    }
}

# =========================================================================
# 4. FUNZIONE DI CALCOLO PROIEZIONI AVANZATA
# =========================================================================

def calcola_proiezioni_avanzate(variazioni_principali, variazioni_aggiuntive):
    """Calcola le proiezioni al 2040 considerando tutti i parametri"""
    BASE = {
        'pil': 34222.0, 'energia': 2298.0, 'vita': 83.4,
        'temperatura': 17.8, 'over65': 24.2, 'spesa': 3283.0
    }
    var_temp = variazioni_principali.get('temperatura_media', 0)
    var_over65 = variazioni_principali.get('over65_percentuale', 0)
    var_spesa = variazioni_principali.get('spesa_sanitaria_pro_capite', 0)
    
    # 1. Calcola nuovo PIL
    pil_impatto = (0.5 * var_spesa) - (0.3 * var_temp) - (0.2 * var_over65)
    for chiave, variazione in variazioni_aggiuntive.items():
        for categoria in [PARAMETRI_DEMOGRAFICI, PARAMETRI_ECONOMICI, PARAMETRI_AMBIENTALI, PARAMETRI_SOCIALI, PARAMETRI_SANITARI_AGGIUNTIVI]:
            if chiave in categoria:
                pil_impatto += categoria[chiave]['impatto_pil'] * variazione
    
    nuovo_pil = BASE['pil'] * (1 + pil_impatto / 100)
    pil_var_perc = ((nuovo_pil - BASE['pil']) / BASE['pil']) * 100
    
    # 2. Calcola nuovo consumo energetico
    energia_impatto = (0.6 * var_temp) - (0.4 * var_over65) + (0.3 * var_spesa) + (0.3 * pil_var_perc)
    for chiave, variazione in variazioni_aggiuntive.items():
        for categoria in [PARAMETRI_DEMOGRAFICI, PARAMETRI_ECONOMICI, PARAMETRI_AMBIENTALI, PARAMETRI_SOCIALI, PARAMETRI_SANITARI_AGGIUNTIVI]:
            if chiave in categoria:
                energia_impatto += categoria[chiave]['impatto_energia'] * variazione
    nuovo_energia = BASE['energia'] * (1 + energia_impatto / 100)
    
    # 3. Calcola nuova aspettativa di vita
    vita_impatto = (0.4 * var_spesa) - (0.2 * var_temp) + (0.15 * var_over65) + (0.1 * pil_var_perc)
    for chiave, variazione in variazioni_aggiuntive.items():
        for categoria in [PARAMETRI_DEMOGRAFICI, PARAMETRI_ECONOMICI, PARAMETRI_AMBIENTALI, PARAMETRI_SOCIALI, PARAMETRI_SANITARI_AGGIUNTIVI]:
            if chiave in categoria:
                vita_impatto += categoria[chiave]['impatto_vita'] * variazione
    nuovo_vita = BASE['vita'] * (1 + vita_impatto / 100)
    
    return {
        'pil_pro_capite': nuovo_pil,
        'consumo_energetico': nuovo_energia,
        'aspettativa_vita': nuovo_vita,
        'temperatura_media': BASE['temperatura'] * (1 + var_temp / 100),
        'over65_percentuale': BASE['over65'] * (1 + var_over65 / 100),
        'spesa_sanitaria_pro_capite': BASE['spesa'] * (1 + var_spesa / 100)
    }

# =========================================================================
# 5. LOGICA DI INIZIALIZZAZIONE DATI (CACHE WEB)
# =========================================================================

@st.cache_data
def ottieni_dati_globali():
    """Inizializza o recupera i dati storici dal database SQLite"""
    if os.path.exists('dati_reali.db'):
        try:
            conn = sqlite3.connect('dati_reali.db')
            df_locale = pd.read_sql_query('SELECT * FROM dati_italia', conn)
            conn.close()
            return df_locale
        except: 
            pass
    
    dati_scaricati = scarica_tutti_dati()
    df_nuovo = crea_dataframe(dati_scaricati)
    if df_nuovo is not None:
        try:
            conn = sqlite3.connect('dati_reali.db')
            df_nuovo.to_sql('dati_italia', conn, if_exists='replace', index=False)
            conn.close()
        except: 
            pass
    return df_nuovo

# Caricamento effettivo del DataFrame globale
df = ottieni_dati_globali()

if df is None or len(df) == 0:
    st.error("❌ Errore: Nessun dato disponibile!")
    st.stop()

# Generazione immediata dei modelli di regressione per il sito web
previsioni_dict = {}
for col in [c for c in df.columns if c != 'anno']:
    modello, prev, r2 = crea_modello_previsione(df, col)
    if modello is not None:
        previsioni_dict[col] = {
            'modello': modello, 'previsioni': prev, 'r2': r2,
            'ultimo_anno': df['anno'].max(), 'ultimo_valore': df[col].iloc[-1]
        }

# =========================================================================
# 6. INTERFACCIA UTENTE CON STREAMLIT
# =========================================================================

# Menu di navigazione sulla barra laterale sinistra
st.sidebar.title("🎮 Navigazione")
pagina = st.sidebar.radio("Seleziona una sezione:", ["Dati Storici", "Modelli Previsionali", "Simulatore Avanzato 2040"])

# SEZIONE 1: DATI STORICI
if pagina == "Dati Storici":
    st.header("📋 Analisi Storica Dati Italia (2000-2025)")
    st.dataframe(df.style.format(precision=2), use_container_width=True)
    
    st.subheader("📈 Grafico Andamento Storico")
    indicatore_scelto = st.selectbox("Scegli la metrica da visualizzare:", [c for c in df.columns if c != 'anno'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['anno'], df[indicatore_scelto], marker='o', color='#1a237e', linewidth=2)
    ax.set_title(f"Andamento storico di {indicatore_scelto.replace('_', ' ').title()}")
    ax.set_xlabel("Anno")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# SEZIONE 2: MODELLI PREVISIONALI
elif pagina == "Modelli Previsionali":
    st.header("🔮 Previsioni Statistiche fino al 2040")
    st.markdown("I grafici mostrano la regressione lineare calcolata sui dati storici passati.")
    
    indicatore_prev = st.selectbox("Seleziona indicatore per la proiezione:", list(previsioni_dict.keys()))
    
    info_prev = previsioni_dict[indicatore_prev]
    anni_passati = df['anno'].values
    valori_passati = df[indicatore_prev].values
    anni_futuri = np.array(list(range(int(info_prev['ultimo_anno']) + 1, 2041)))
    valori_futuri = info_prev['previsioni']
    
    col1, col2 = st.columns(2)
    col1.metric("Affidabilità Modello (R²)", f"{info_prev['r2']:.4f}")
    col2.metric("Valore Atteso al 2040", f"{valori_futuri[-1]:.2f}")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(anni_passati, valori_passati, label='Dati Storici Reali', color='#1a237e', marker='o')
    ax.plot(anni_futuri, valori_futuri, label='Proiezione Lineare', color='#c62828', linestyle='--')
    ax.axvline(x=info_prev['ultimo_anno'], color='gray', linestyle=':', alpha=0.7)
    ax.set_title(f"Modello di Regressione per {indicatore_prev.replace('_', ' ').title()}")
    ax.set_xlabel("Anno")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# SEZIONE 3: SIMULATORE AVANZATO
elif pagina == "Simulatore Avanzato 2040":
    st.header("🎛️ Simulatore di Scenari Futuri")
    st.markdown("Modifica i parametri per simulare variazioni percentuali aggregate e osservare gli impatti macroeconomici stimati al 2040.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Variazioni Principali (%)")
    v_temp = st.sidebar.slider("Temperatura Media", -50.0, 50.0, 0.0, step=1.0)
    v_over = st.sidebar.slider("Popolazione Over 65", -50.0, 50.0, 0.0, step=1.0)
    v_spesa = st.sidebar.slider("Spesa Sanitaria", -50.0, 50.0, 0.0, step=1.0)
    
    var_principali = {
        'temperatura_media': v_temp,
        'over65_percentuale': v_over,
        'spesa_sanitaria_pro_capite': v_spesa
    }
    
    var_aggiuntive = {}
    
    # Divisione in Tab degli slider per mantenere l'interfaccia scansionabile ed elegante
    st.subheader("Modifica i fattori di impatto secondari")
    tab1, tab2, tab3, tab4 = st.tabs(["Demografia & Economia", "Ambiente", "Sociale", "Sanità"])
    
    with tab1:
        for k, v in PARAMETRI_DEMOGRAFICI.items():
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, 0.0, help=v['descrizione'])
        for k, v in PARAMETRI_ECONOMICI.items():
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, 0.0, help=v['descrizione'])
            
    with tab2:
        for k, v in PARAMETRI_AMBIENTALI.items():
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, 0.0, help=v['descrizione'])
            
    with tab3:
        for k, v in PARAMETRI_SOCIALI.items():
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, 0.0, help=v['descrizione'])
            
    with tab4:
        for k, v in PARAMETRI_SANITARI_AGGIUNTIVI.items():
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, 0.0, help=v['descrizione'])
            
    # Calcolo dinamico in tempo reale ad ogni movimento degli slider
    risultati = calcola_proiezioni_avanzate(var_principali, var_aggiuntive)
    
    st.markdown("### 📊 Risultati della Simulazione al 2040")
    
    # Visualizzazione delle metriche aggregate in colonne distinte
    c1, c2, c3 = st.columns(3)
    c1.metric("PIL Pro Capite Stimato", f"${risultati['pil_pro_capite']:.2f}")
    c2.metric("Consumo Energetico Stimato", f"{risultati['consumo_energetico']:.1f} kg OE")
    c3.metric("Aspettativa di Vita Stimata", f"{risultati['aspettativa_vita']:.2f} Anni")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Temperatura Media Proiettata", f"{risultati['temperatura_media']:.2f} °C")
    c5.metric("Popolazione Over 65 Proiettata", f"{risultati['over65_percentuale']:.2f} %")
    c6.metric("Spesa Sanitaria Proiettata", f"${risultati['spesa_sanitaria_pro_capite']:.2f}")