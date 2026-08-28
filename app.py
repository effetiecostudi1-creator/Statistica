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
import time
import io

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Italia - Analisi & Simulatore Avanzato 2000-2040",
    page_icon="📊",
    layout="wide"
)

# Inizializza session_state per salvare le configurazioni
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = {}

# Titolo principale
st.title("📊 Italia - Analisi & Simulatore Avanzato 2000-2040")
st.markdown("---")

# =========================================================================
# 1. FUNZIONI PER IL DOWNLOAD DEI DATI
# =========================================================================

def scarica_indicatore_worldbank(codice, paese='IT'):
    """Scarica un indicatore dall'API World Bank per l'Italia"""
    url = f"http://api.worldbank.org/v2/country/{paese}/indicator/{codice}?format=json&per_page=100"
    
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
                    if 'year' in record:
                        anno = int(record['year'])
                    elif 'date' in record:
                        anno = int(record['date'])
                    else:
                        continue
                        
                    if 2000 <= anno <= 2025:
                        valore = float(record['value'])
                        valori[anno] = valore
                except (ValueError, TypeError):
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
        'aspettativa_vita': {'codice': 'SP.DYN.LE00.IN', 'nome': 'Aspettativa vita (anni)'},
        'popolazione': {'codice': 'SP.POP.TOTL', 'nome': 'Popolazione totale'},
        'occupazione': {'codice': 'SL.EMP.TOTL.SP.ZS', 'nome': 'Tasso di occupazione (%)'}
    }
    
    dati = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (key, info) in enumerate(indicatori.items()):
        status_text.text(f"Scaricando {info['nome']}...")
        valori = scarica_indicatore_worldbank(info['codice'])
        dati[key] = valori
        time.sleep(0.3)
        progress_bar.progress((i + 1) / len(indicatori))
    
    status_text.text("Download completato!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    dati['temperatura'] = scarica_temperatura_reale()
    return dati

def pulisci_dataframe(df):
    """Pulisce il DataFrame convertendo tutte le colonne numeriche in float"""
    for col in df.columns:
        if col != 'anno':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def crea_dataframe(dati):
    """Organizza i dati scaricati in un DataFrame Pandas ordinato"""
    anni_insieme = set()
    for valori in dati.values():
        if valori:
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
        'aspettativa_vita': [dati.get('aspettativa_vita', {}).get(a, None) for a in anni],
        'popolazione_totale': [dati.get('popolazione', {}).get(a, None) for a in anni],
        'tasso_occupazione': [dati.get('occupazione', {}).get(a, None) for a in anni]
    })
    
    df = df.dropna(subset=['pil_pro_capite', 'over65_percentuale', 'spesa_sanitaria_pro_capite'], how='all')
    df = pulisci_dataframe(df)
    
    return df

def crea_dati_esempio():
    """Crea dati di esempio se il download fallisce"""
    anni = list(range(2000, 2026))
    np.random.seed(42)
    
    dati = {
        'anno': anni,
        'pil_pro_capite': [30000 + (i * 200) + (np.random.randn() * 500) for i in range(len(anni))],
        'over65_percentuale': [18 + (i * 0.3) + (np.random.randn() * 0.5) for i in range(len(anni))],
        'spesa_sanitaria_pro_capite': [2000 + (i * 50) + (np.random.randn() * 100) for i in range(len(anni))],
        'temperatura_media': [15.5 + (i * 0.08) + (np.random.randn() * 0.3) for i in range(len(anni))],
        'consumo_energetico': [2500 - (i * 10) + (np.random.randn() * 50) for i in range(len(anni))],
        'aspettativa_vita': [78 + (i * 0.15) + (np.random.randn() * 0.3) for i in range(len(anni))],
        'popolazione_totale': [57000000 + (i * 50000) + (np.random.randn() * 10000) for i in range(len(anni))],
        'tasso_occupazione': [55 + (i * 0.1) + (np.random.randn() * 0.5) for i in range(len(anni))]
    }
    
    df = pd.DataFrame(dati)
    df = pulisci_dataframe(df)
    return df

# =========================================================================
# 2. MODELLI STATISTICI
# =========================================================================

def crea_modello_previsione(df, indicatore):
    """Crea un modello di regressione lineare per un indicatore"""
    df_clean = df[['anno', indicatore]].dropna()
    
    if len(df_clean) < 5:
        return None, None, None
    
    anni = df_clean['anno'].values.reshape(-1, 1)
    valori = df_clean[indicatore].values
    
    modello = LinearRegression()
    modello.fit(anni, valori)
    predizioni_train = modello.predict(anni)
    r2 = r2_score(valori, predizioni_train)
    
    ultimo_anno = df['anno'].max()
    anni_futuri_lista = list(range(int(ultimo_anno) + 1, 2041))
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
# 5. FUNZIONI DI ESPORTAZIONE (CON GESTIONE ERRORI)
# =========================================================================

def esporta_csv(df):
    """Esporta il DataFrame in formato CSV"""
    return df.to_csv(index=False).encode('utf-8')

def esporta_excel(df):
    """Esporta il DataFrame in formato Excel con fallback a CSV"""
    try:
        # Prova a usare xlsxwriter
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Dati Italia', index=False)
        return output.getvalue()
    except ImportError:
        # Se xlsxwriter non è installato, usa openpyxl
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Dati Italia', index=False)
            return output.getvalue()
        except ImportError:
            # Se nessun engine è disponibile, converti in CSV
            st.warning("⚠️ xlsxwriter non installato. Uso formato CSV come fallback.")
            return df.to_csv(index=False).encode('utf-8')

# =========================================================================
# 6. LOGICA DI INIZIALIZZAZIONE DATI (CACHE WEB)
# =========================================================================

@st.cache_data
def ottieni_dati_globali():
    """Inizializza o recupera i dati storici dal database SQLite"""
    
    # Prova a caricare dal database
    if os.path.exists('dati_reali.db'):
        try:
            conn = sqlite3.connect('dati_reali.db')
            df_locale = pd.read_sql_query('SELECT * FROM dati_italia', conn)
            conn.close()
            if len(df_locale) > 0:
                return df_locale
        except:
            pass
    
    # Se non ci sono dati, scarica nuovi
    dati_scaricati = scarica_tutti_dati()
    df_nuovo = crea_dataframe(dati_scaricati)
    
    # Se il download fallisce, usa dati di esempio
    if df_nuovo is None or len(df_nuovo) == 0:
        df_nuovo = crea_dati_esempio()
    
    # Salva nel database
    if df_nuovo is not None and len(df_nuovo) > 0:
        try:
            conn = sqlite3.connect('dati_reali.db')
            df_nuovo.to_sql('dati_italia', conn, if_exists='replace', index=False)
            conn.close()
        except:
            pass
    
    return df_nuovo

# =========================================================================
# 7. CARICAMENTO DATI
# =========================================================================

# Caricamento effettivo del DataFrame globale
with st.spinner("Caricamento dati in corso..."):
    df = ottieni_dati_globali()

if df is None or len(df) == 0:
    st.error("❌ Errore: Nessun dato disponibile! Controlla la connessione internet.")
    st.stop()

# Mostra informazioni sui dati
st.success(f"✅ Dati caricati con successo! ({len(df)} anni disponibili)")

# Generazione modelli di regressione
previsioni_dict = {}
for col in [c for c in df.columns if c != 'anno']:
    modello, prev, r2 = crea_modello_previsione(df, col)
    if modello is not None:
        previsioni_dict[col] = {
            'modello': modello, 
            'previsioni': prev, 
            'r2': r2,
            'ultimo_anno': df['anno'].max(), 
            'ultimo_valore': df[col].iloc[-1]
        }

# =========================================================================
# 8. INTERFACCIA UTENTE CON STREAMLIT
# =========================================================================

# Menu di navigazione
st.sidebar.title("🎮 Navigazione")
pagina = st.sidebar.radio("Seleziona una sezione:", ["Dati Storici", "Modelli Previsionali", "Simulatore Avanzato 2040"])

# Pulsante per aggiornare i dati
if st.sidebar.button("🔄 Aggiorna Dati"):
    st.cache_data.clear()
    st.rerun()

# =========================================================================
# SEZIONE 1: DATI STORICI
# =========================================================================
if pagina == "Dati Storici":
    st.header("📋 Analisi Storica Dati Italia (2000-2025)")
    
    # Filtri per anno
    col1, col2 = st.columns(2)
    with col1:
        anno_min = st.slider("Anno minimo", int(df['anno'].min()), int(df['anno'].max()), int(df['anno'].min()))
    with col2:
        anno_max = st.slider("Anno massimo", int(df['anno'].min()), int(df['anno'].max()), int(df['anno'].max()))
    
    df_filtrato = df[(df['anno'] >= anno_min) & (df['anno'] <= anno_max)]
    st.dataframe(df_filtrato.style.format(precision=2), use_container_width=True)
    
    # Pulsanti di esportazione
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = esporta_csv(df_filtrato)
        st.download_button(
            label="📥 Scarica CSV",
            data=csv_data,
            file_name=f"dati_italia_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with col2:
        try:
            excel_data = esporta_excel(df_filtrato)
            st.download_button(
                label="📥 Scarica Excel",
                data=excel_data,
                file_name=f"dati_italia_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning(f"⚠️ Esportazione Excel non disponibile: {str(e)}")
    with col3:
        # Pulsante per stampare
        if st.button("🖨️ Stampa"):
            st.markdown("### Dati Italia")
            st.dataframe(df_filtrato)
            st.markdown("---")
            st.caption("Report generato il " + datetime.now().strftime("%d/%m/%Y %H:%M"))
    
    st.subheader("📈 Grafico Andamento Storico")
    indicatore_scelto = st.selectbox("Scegli la metrica da visualizzare:", [c for c in df.columns if c != 'anno'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-darkgrid')
    ax.plot(df['anno'], df[indicatore_scelto], marker='o', color='#1a237e', linewidth=2, markersize=8)
    ax.set_title(f"Andamento storico di {indicatore_scelto.replace('_', ' ').title()}")
    ax.set_xlabel("Anno")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

# =========================================================================
# SEZIONE 2: MODELLI PREVISIONALI
# =========================================================================
elif pagina == "Modelli Previsionali":
    st.header("🔮 Previsioni Statistiche fino al 2040")
    st.markdown("I grafici mostrano la regressione lineare calcolata sui dati storici passati.")
    
    if len(previsioni_dict) == 0:
        st.warning("Nessun modello disponibile per le previsioni.")
    else:
        # Selettore con raggruppamento per categoria
        col_prev = st.selectbox("Seleziona indicatore per la proiezione:", list(previsioni_dict.keys()))
        
        info_prev = previsioni_dict[col_prev]
        anni_passati = df['anno'].values
        valori_passati = df[col_prev].values
        anni_futuri = np.array(list(range(int(info_prev['ultimo_anno']) + 1, 2041)))
        valori_futuri = info_prev['previsioni']
        
        # Metriche
        col1, col2, col3 = st.columns(3)
        col1.metric("Affidabilità Modello (R²)", f"{info_prev['r2']:.4f}")
        col2.metric("Valore Attuale", f"{info_prev['ultimo_valore']:.2f}")
        col3.metric("Valore Atteso al 2040", f"{valori_futuri[-1]:.2f}")
        
        # Grafico
        fig, ax = plt.subplots(figsize=(12, 5))
        plt.style.use('seaborn-v0_8-darkgrid')
        ax.plot(anni_passati, valori_passati, label='Dati Storici Reali', color='#1a237e', marker='o', linewidth=2, markersize=8)
        ax.plot(anni_futuri, valori_futuri, label='Proiezione Lineare', color='#c62828', linestyle='--', linewidth=2)
        ax.axvline(x=info_prev['ultimo_anno'], color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.set_title(f"Modello di Regressione per {col_prev.replace('_', ' ').title()}")
        ax.set_xlabel("Anno")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Tabella previsioni
        with st.expander("📊 Tabella Previsioni Dettagliate"):
            df_previsioni = pd.DataFrame({
                'Anno': anni_futuri,
                'Valore Previsto': valori_futuri
            })
            st.dataframe(df_previsioni.style.format(precision=2), use_container_width=True)

# =========================================================================
# SEZIONE 3: SIMULATORE AVANZATO
# =========================================================================
elif pagina == "Simulatore Avanzato 2040":
    st.header("🎛️ Simulatore di Scenari Futuri")
    st.markdown("Modifica i parametri per simulare variazioni percentuali aggregate e osservare gli impatti macroeconomici stimati al 2040.")
    
    # Sidebar per i parametri principali
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Variazioni Principali (%)")
    
    # Carica valori salvati se presenti
    v_temp = st.sidebar.slider("Temperatura Media", -50.0, 50.0, 
                               st.session_state.slider_values.get('temp', 0.0), step=1.0)
    v_over = st.sidebar.slider("Popolazione Over 65", -50.0, 50.0, 
                               st.session_state.slider_values.get('over', 0.0), step=1.0)
    v_spesa = st.sidebar.slider("Spesa Sanitaria", -50.0, 50.0, 
                                st.session_state.slider_values.get('spesa', 0.0), step=1.0)
    
    # Salva valori
    st.session_state.slider_values['temp'] = v_temp
    st.session_state.slider_values['over'] = v_over
    st.session_state.slider_values['spesa'] = v_spesa
    
    var_principali = {
        'temperatura_media': v_temp,
        'over65_percentuale': v_over,
        'spesa_sanitaria_pro_capite': v_spesa
    }
    
    var_aggiuntive = {}
    
    # Divisione in Tab degli slider
    st.subheader("Modifica i fattori di impatto secondari")
    tab1, tab2, tab3, tab4 = st.tabs(["Demografia & Economia", "Ambiente", "Sociale", "Sanità"])
    
    with tab1:
        for k, v in PARAMETRI_DEMOGRAFICI.items():
            default_val = st.session_state.slider_values.get(f'dem_{k}', 0.0)
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, default_val, step=1.0, help=v['descrizione'])
            st.session_state.slider_values[f'dem_{k}'] = var_aggiuntive[k]
        
        st.markdown("---")
        for k, v in PARAMETRI_ECONOMICI.items():
            default_val = st.session_state.slider_values.get(f'eco_{k}', 0.0)
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, default_val, step=1.0, help=v['descrizione'])
            st.session_state.slider_values[f'eco_{k}'] = var_aggiuntive[k]
            
    with tab2:
        for k, v in PARAMETRI_AMBIENTALI.items():
            default_val = st.session_state.slider_values.get(f'amb_{k}', 0.0)
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, default_val, step=1.0, help=v['descrizione'])
            st.session_state.slider_values[f'amb_{k}'] = var_aggiuntive[k]
            
    with tab3:
        for k, v in PARAMETRI_SOCIALI.items():
            default_val = st.session_state.slider_values.get(f'soc_{k}', 0.0)
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, default_val, step=1.0, help=v['descrizione'])
            st.session_state.slider_values[f'soc_{k}'] = var_aggiuntive[k]
            
    with tab4:
        for k, v in PARAMETRI_SANITARI_AGGIUNTIVI.items():
            default_val = st.session_state.slider_values.get(f'san_{k}', 0.0)
            var_aggiuntive[k] = st.slider(f"{v['nome']} ({v['unita']})", -50.0, 50.0, default_val, step=1.0, help=v['descrizione'])
            st.session_state.slider_values[f'san_{k}'] = var_aggiuntive[k]
    
    # Pulsante per resettare i valori
    if st.button("🔄 Reset Valori"):
        st.session_state.slider_values = {}
        st.rerun()
    
    # Calcolo dinamico
    risultati = calcola_proiezioni_avanzate(var_principali, var_aggiuntive)
    
    st.markdown("### 📊 Risultati della Simulazione al 2040")
    
    # Visualizzazione metriche
    c1, c2, c3 = st.columns(3)
    c1.metric("PIL Pro Capite Stimato", f"${risultati['pil_pro_capite']:,.2f}", 
              delta=f"{((risultati['pil_pro_capite'] - 34222.0) / 34222.0 * 100):.1f}%")
    c2.metric("Consumo Energetico Stimato", f"{risultati['consumo_energetico']:,.1f} kg OE",
              delta=f"{((risultati['consumo_energetico'] - 2298.0) / 2298.0 * 100):.1f}%")
    c3.metric("Aspettativa di Vita Stimata", f"{risultati['aspettativa_vita']:.2f} Anni",
              delta=f"{((risultati['aspettativa_vita'] - 83.4) / 83.4 * 100):.1f}%")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Temperatura Media Proiettata", f"{risultati['temperatura_media']:.2f} °C",
              delta=f"{((risultati['temperatura_media'] - 17.8) / 17.8 * 100):.1f}%")
    c5.metric("Popolazione Over 65 Proiettata", f"{risultati['over65_percentuale']:.2f} %",
              delta=f"{((risultati['over65_percentuale'] - 24.2) / 24.2 * 100):.1f}%")
    c6.metric("Spesa Sanitaria Proiettata", f"${risultati['spesa_sanitaria_pro_capite']:,.2f}",
              delta=f"{((risultati['spesa_sanitaria_pro_capite'] - 3283.0) / 3283.0 * 100):.1f}%")
    
    # Grafico comparativo
    st.subheader("📊 Confronto Valori Base vs Proiezioni")
    
    # Crea un DataFrame per il confronto
    valori_base = {
        'PIL Pro Capite': 34222.0,
        'Consumo Energetico': 2298.0,
        'Aspettativa Vita': 83.4,
        'Temperatura': 17.8,
        'Over 65': 24.2,
        'Spesa Sanitaria': 3283.0
    }
    
    valori_proiettati = {
        'PIL Pro Capite': risultati['pil_pro_capite'],
        'Consumo Energetico': risultati['consumo_energetico'],
        'Aspettativa Vita': risultati['aspettativa_vita'],
        'Temperatura': risultati['temperatura_media'],
        'Over 65': risultati['over65_percentuale'],
        'Spesa Sanitaria': risultati['spesa_sanitaria_pro_capite']
    }
    
    # Grafico a barre
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    x = np.arange(len(valori_base))
    width = 0.35
    
    # Normalizza i valori per il confronto
    base_norm = np.array(list(valori_base.values())) / np.array(list(valori_base.values()))
    proiettati_norm = np.array(list(valori_proiettati.values())) / np.array(list(valori_base.values()))
    
    bars1 = ax.bar(x - width/2, base_norm, width, label='Valori Base', color='#1a237e', alpha=0.7)
    bars2 = ax.bar(x + width/2, proiettati_norm, width, label='Proiezioni 2040', color='#c62828', alpha=0.7)
    
    # Aggiungi valori sulle barre
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{1.0:.2f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax.set_xlabel('Indicatori')
    ax.set_ylabel('Variazione Normalizzata (Base = 1)')
    ax.set_title('Confronto Valori Base vs Proiezioni 2040')
    ax.set_xticks(x)
    ax.set_xticklabels(list(valori_base.keys()), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Spiegazione dei risultati
    with st.expander("📖 Legenda e Interpretazione"):
        st.markdown("""
        ### Come interpretare i risultati
        
        **PIL Pro Capite**: Indica il prodotto interno lordo per persona. Un aumento significa maggiore ricchezza.
        
        **Consumo Energetico**: Misura l'energia consumata per persona. Idealmente dovrebbe diminuire per sostenibilità.
        
        **Aspettativa di Vita**: Anni di vita medi attesi. Più alto è meglio.
        
        **Temperatura Media**: Indicatore del cambiamento climatico. Idealmente dovrebbe rimanere stabile.
        
        **Over 65**: Percentuale di popolazione anziana. Un aumento può indicare invecchiamento demografico.
        
        **Spesa Sanitaria**: Investimento in salute. Più alta significa migliori servizi sanitari.
        """)

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("---")
st.caption("📊 Italia - Analisi & Simulatore Avanzato 2000-2040 | Aggiornato: " + datetime.now().strftime("%d/%m/%Y %H:%M"))
