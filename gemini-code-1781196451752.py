# ==============================================================================
# 📱 ULTIMATE AI V14.8 - DATOS REALES (ÚLTIMOS 2 AÑOS / RACHA 10 PARTIDOS)
# ==============================================================================

import streamlit as st
import streamlit.components.v1 as components  
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Predicciones", page_icon="🏆", layout="centered")

# --- FUNCIÓN: ESTADO DE FORMA OFICIAL (ÚLTIMOS 5 PARTIDOS) ---
def get_form_oficial(t, df_historico, fecha_partido):
    try:
        pasado = df_historico[
            ((df_historico['home_team'] == t) | (df_historico['away_team'] == t)) & 
            (df_historico['date'] < fecha_partido) &
            (df_historico['tournament'] != 'Friendly')
        ].tail(5)
        
        puntos = 0
        for _, row in pasado.iterrows():
            if row['home_team'] == t and row['home_score'] > row['away_score']: puntos += 3
            elif row['away_team'] == t and row['away_score'] > row['home_score']: puntos += 3
            elif row['home_score'] == row['away_score']: puntos += 1
        return puntos
    except:
        return 0

@st.cache_data(show_spinner=False)
def cargar_y_enriquecer_selecciones():
    if not os.path.exists("datos_reales_selecciones.csv"):
        return None
        
    df = pd.read_csv("datos_reales_selecciones.csv")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    np.random.seed(42)
    equipos_unicos = pd.concat([df['home_team'], df['away_team']]).unique()
    simulacion_ranking = {eq: np.random.randint(1, 150) for eq in equipos_unicos}
    df['home_rank'] = df['home_team'].map(simulacion_ranking)
    df['away_rank'] = df['away_team'].map(simulacion_ranking)
    
    condiciones = [(df['home_score'] > df['away_score']), (df['home_score'] < df['away_score'])]
    df['FTR'] = np.select(condiciones, ['H', 'A'], default='D')
    return df.sort_values('date').reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def entrenar_ia(_df):
    X_list, targets = [], {'res':[], 'g_h':[], 'g_a':[], 'c_h':[], 'c_a':[], 's_h':[], 's_a':[], 'st_h':[], 'st_a':[], 't_h':[], 't_a':[]}
    equipos = pd.concat([_df['home_team'], _df['away_team']]).unique()
    team_stats = {eq: {'GF': [], 'GC': [], 'S': [], 'ST': [], 'C': [], 'Y': [], 'Pts': []} for eq in equipos}
    
    for idx, row in _df.iterrows():
        h, a = row['home_team'], row['away_team']
        
        def get_avg(t, stat):
            arr = team_stats[t][stat][-10:]
            return np.mean(arr) if len(arr) > 0 else 0.0
            
        def get_form(t): 
            return sum(team_stats[t]['Pts'][-10:])
            
        X_list.append({
            'H_GF': get_avg(h, 'GF'), 'H_GC': get_avg(h, 'GC'), 'H_S': get_avg(h, 'S'), 
            'H_ST': get_avg(h, 'ST'), 'H_C': get_avg(h, 'C'), 'H_Y': get_avg(h, 'Y'), 'H_Form': get_form(h),
            'A_GF': get_avg(a, 'GF'), 'A_GC': get_avg(a, 'GC'), 'A_S': get_avg(a, 'S'), 
            'A_ST': get_avg(a, 'ST'), 'A_C': get_avg(a, 'C'), 'A_Y': get_avg(a, 'Y'), 'A_Form': get_form(a),
            'Neutral': 1 if row.get('neutral', False) else 0,
            'H_Rank': row['home_rank'], 'A_Rank': row['away_rank'],
            'Rank_Diff': row['away_rank'] - row['home_rank'],
            'H_Form_Official': get_form_oficial(h, _df, row['date']),
            'A_Form_Official': get_form_oficial(a, _df, row['date']),
            'Is_Qualifier': 1 if 'qualification' in str(row['tournament']).lower() else 0
        })
        
        targets['res'].append(row['FTR'])
        targets['g_h'].append(row['home_score']); targets['g_a'].append(row['away_score'])
        targets['c_h'].append(row['HC']); targets['c_a'].append(row['AC'])
        targets['s_h'].append(row['HS']); targets['s_a'].append(row['AS'])
        targets['st_h'].append(row['HST']); targets['st_a'].append(row['AST'])
        targets['t_h'].append(row['HY']); targets['t_a'].append(row['AY'])
        
        pts_h = 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)
        pts_a = 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)
        
        team_stats[h]['Pts'].append(pts_h); team_stats[a]['Pts'].append(pts_a)
        team_stats[h]['GF'].append(row['home_score']); team_stats[a]['GF'].append(row['away_score'])
        team_stats[h]['GC'].append(row['away_score']); team_stats[a]['GC'].append(row['home_score'])
        team_stats[h]['S'].append(row['HS']); team_stats[a]['S'].append(row['AS'])
        team_stats[h]['ST'].append(row['HST']); team_stats[a]['ST'].append(row['AST'])
        team_stats[h]['C'].append(row['HC']); team_stats[a]['C'].append(row['AC'])
        team_stats[h]['Y'].append(row['HY']); team_stats[a]['Y'].append(row['AY'])

    X = pd.DataFrame(X_list).fillna(0)
    le = LabelEncoder()
    y_res = le.fit_transform(targets['res']) 
    
    rs = 42 
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=rs).fit(X, y_res)
    h_mods = {
        'gol': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['g_h']),
        'corn': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['c_h']),
        'shot': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['s_h']),
        'shot_t': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['st_h']),
        'card': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['t_h'])
    }
    a_mods = {
        'gol': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['g_a']),
        'corn': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['c_a']),
        'shot': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['s_a']),
        'shot_t': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['st_a']),
        'card': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['t_a'])
    }
    return clf, le, h_mods, a_mods, team_stats

st.title("🏆 ULTIMATE AI V14.8")
st.markdown("### Predicciones IA (By L.Niquén)")

df_global = cargar_y_enriquecer_selecciones()

if df_global is None:
    st.error("⚠️ Faltan datos: Primero debes ejecutar el bot 'descargar_datos.py' para generar el archivo 'datos_reales_selecciones.csv'.")
    st.stop()

with st.spinner('Entrenando IA con datos reales...'):
    clf, le, h_mods, a_mods, stats = entrenar_ia(df_global)

equipos_activos = sorted([eq for eq in stats.keys() if len(stats[eq]['Pts']) > 0])

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("🌍 Equipo 1 (Local):", equipos_activos, index=0)
with col2:
    away = st.selectbox("🌍 Equipo 2 (Visita):", equipos_activos, index=1 if len(equipos_activos)>1 else 0)

col_opt1, col_opt2 = st.columns(2)
with col_opt1: is_neutral = st.checkbox("Cancha Neutral", value=True)
with col_opt2: is_qualifier = st.checkbox("Partido Oficial", value=True)

# ==============================================================================
# 🛡️ SELECTOR DE CONTEXTO Y LÓGICA DE LA IA
# ==============================================================================
st.markdown("---")
contexto_partido = st.radio(
    "🛡️ Contexto del Partido (Nivel de Riesgo):",
    [
        "Fase de Grupos / Amistoso (Juego más abierto y ofensivo)", 
        "Eliminación Directa / Final (Matar o morir, alta cautela táctica)"
    ]
)

# Lógica de mercados según el contexto seleccionado
instrucciones_mercado = ""

if "Eliminación" in contexto_partido:
    instrucciones_mercado = """
    🛡️ MERCADOS PARA TORNEOS DE ELIMINACIÓN (ALTA CAUTELA):
    Al analizar este partido, tu deber es enfocar la estrategia en la prevención de riesgos y justificar tu predicción basándote en estos mercados:
    * Menos de 2.5 Goles (Under 2.5): Es el rey de las fases finales. Las defensas se cierran, las líneas se juntan y nadie regala un centímetro. Si detectas que ambos equipos tienen defensas sólidas, el Under es una decisión estadísticamente muy inteligente.
    * Empate en el Primer Tiempo (X - Mitad 1): En partidos donde la eliminación está en juego, los primeros 45 minutos suelen ser de puro estudio, contención y respeto mutuo. Nadie quiere arriesgar su capital táctico temprano.
    * Más de X Tarjetas (Over Tarjetas): Cuando el miedo a recibir un gol es tan grande, los equipos prefieren cortar cualquier contragolpe peligroso con "faltas tácticas". Ese juego friccionado y preventivo dispara la cantidad de tarjetas amarillas.
    * Clasifica (To Qualify): En lugar de apostar a quién gana en los 90 minutos (donde un empate 0-0 te hace perder la apuesta), el mercado de "Clasifica" te cubre. No importa si la selección gana en el tiempo regular, en la prórroga o por penales; si pasan de ronda, se cobra.
    """
else:
    instrucciones_mercado = """
    ⚽ MERCADOS PARA FASE DE GRUPOS / LIGA / AMISTOSOS:
    Evalúa el terreno basándote estrictamente en estos mercados estadísticamente fiables:
    1. Más / Menos Goles (Over / Under 1.5 o 2.5): Este es el mercado rey. Predecir quién ganará puede arruinarse por una tarjeta roja o un penal, pero el flujo del partido es muy predecible. Calcula con altísima precisión si el partido terminará con 2 o más goles cruzando los promedios ofensivos vs debilidades defensivas (xG, tiros). Ideal para clara diferencia de niveles o defensas frágiles.
    2. Apuesta Sin Empate (Draw No Bet / Empate No Acción): Los empates son increíblemente comunes. Identificar el riesgo requiere medidas de prevención, similar a estructurar una matriz de riesgos. Si el peligro de un empate es alto, este mercado actúa como protección. Ideal para partidos de visitante donde un equipo es superior, pero factores externos equilibran la balanza.
    3. Ambos Equipos Anotan (BTTS - Sí / No): Elimina la necesidad de adivinar quién se llevará
