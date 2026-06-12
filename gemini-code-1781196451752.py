# ==============================================================================
# 📱 ULTIMATE AI V15.1 - DATOS REALES (SIMETRÍA ABSOLUTA CORREGIDA)
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

# --- FUNCIÓN: ESTADO DE FORMA OFICIAL ---
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

st.title("🏆 ULTIMATE AI V15.1")
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
    home = st.selectbox("🌍 Equipo 1 (Local / Izquierda):", equipos_activos, index=0)
with col2:
    away = st.selectbox("🌍 Equipo 2 (Visita / Derecha):", equipos_activos, index=1 if len(equipos_activos)>1 else 0)

col_opt1, col_opt2 = st.columns(2)
with col_opt1: is_neutral = st.checkbox("Cancha Neutral", value=True)
with col_opt2: is_qualifier = st.checkbox("Partido Oficial", value=True)

st.markdown("---")
contexto_partido = st.radio(
    "🛡️ Contexto del Partido (Nivel de Riesgo):",
    [
        "Fase de Grupos / Amistoso (Juego más abierto y ofensivo)", 
        "Eliminación Directa / Final (Matar o morir, alta cautela táctica)"
    ]
)

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
    3. Ambos Equipos Anotan (BTTS - Sí / No): Elimina la necesidad de adivinar quién se llevará los 3 puntos. Solo pregúntate: ¿Ambos tienen capacidad de hacerse daño? Busca patrones de ida y vuelta constante. Ideal para equipos con gran poderío en la delantera pero estadísticas pobres para mantener su portería a cero.
    """

restricciones_mercado = """
🚫 MERCADOS QUE DEBES EVITAR (Alta varianza):
* Marcador Exacto: Es una lotería. Matemáticamente es casi imposible de predecir de forma consistente.
* Primer equipo en anotar: Depende demasiado de factores aleatorios iniciales.
Nunca recomiendes estos dos mercados en tu análisis final.
"""

prompt_ia = f"""
Eres un analista deportivo experto en estadística avanzada.
Analiza el enfrentamiento entre {home} (Local) y {away} (Visitante).

{instrucciones_mercado}

{restricciones_mercado}

Basado en los datos y estadísticas proporcionadas, dame las probabilidades y tu mejor recomendación estructurada para este partido.
"""

if st.button("🚀 GENERAR INFORME", use_container_width=True):
    if home == away:
        st.error("⚠️ Error: Selecciona equipos distintos.")
    else:
        def get_avg_predict(t_data, stat):
            arr = t_data[stat][-10:]
            return np.mean(arr) if len(arr) > 0 else 0.0
            
        def get_form_predict(t_data): 
            return sum(t_data['Pts'][-10:])
            
        fecha_actual = pd.Timestamp.now()
        
        # --- FUNCIÓN MAESTRA PARA CONSTRUIR INPUTS SIN ERRORES ---
        def generar_input_ia(t_local, t_visita):
            h_data, a_data = stats[t_local], stats[t_visita]
            
            hr = df_global[df_global['home_team'] == t_local]['home_rank'].iloc[-1] if not df_global[df_global['home_team'] == t_local].empty else 100
            ar = df_global[df_global['away_team'] == t_visita]['away_rank'].iloc[-1] if not df_global[df_global['away_team'] == t_visita].empty else 100
            
            return pd.DataFrame([{
                'H_GF': get_avg_predict(h_data, 'GF'), 'H_GC': get_avg_predict(h_data, 'GC'), 'H_S': get_avg_predict(h_data, 'S'), 
                'H_ST': get_avg_predict(h_data, 'ST'), 'H_C': get_avg_predict(h_data, 'C'), 'H_Y': get_avg_predict(h_data, 'Y'), 'H_Form': get_form_predict(h_data),
                'A_GF': get_avg_predict(a_data, 'GF'), 'A_GC': get_avg_predict(a_data, 'GC'), 'A_S': get_avg_predict(a_data, 'S'), 
                'A_ST': get_avg_predict(a_data, 'ST'), 'A_C': get_avg_predict(a_data, 'C'), 'A_Y': get_avg_predict(a_data, 'Y'), 'A_Form': get_form_predict(a_data),
                'Neutral': 1 if is_neutral else 0,
                'H_Rank': hr, 'A_Rank': ar, 'Rank_Diff': ar - hr,
                'H_Form_Official': get_form_oficial(t_local, df_global, fecha_actual),
                'A_Form_Official': get_form_oficial(t_visita, df_global, fecha_actual),
                'Is_Qualifier': 1 if is_qualifier else 0
            }])

        # 1. EVALUACIÓN NORMAL (Como dicta la pantalla)
        input_n = generar_input_ia(home, away)
        probs_n = clf.predict_proba(input_n)[0]
        cls = le.inverse_transform(clf.classes_)
        pmap_n = {c: p for c, p in zip(cls, probs_n)}
        
        # 2. LÓGICA DE ESPEJO (TEST-TIME AUGMENTATION PURA)
        if is_neutral:
            input_i = generar_input_ia(away, home)
            probs_i = clf.predict_proba(input_i)[0]
            pmap_i = {c: p for c, p in zip(cls, probs_i)}
            
            p_h = (pmap_n.get('H', 0) + pmap_i.get('A', 0)) / 2
            p_a = (pmap_n.get('A', 0) + pmap_i.get('H', 0)) / 2
            p_d = (pmap_n.get('D', 0) + pmap_i.get('D', 0)) / 2
            
            # Normalización estricta al 100%
            tot_p = p_h + p_a + p_d
            p_h, p_a, p_d = p_h/tot_p, p_a/tot_p, p_d/tot_p
            
            xg_h = (max(0, h_mods['gol'].predict(input_n)[0]) + max(0, a_mods['gol'].predict(input_i)[0])) / 2
            xg_a = (max(0, a_mods['gol'].predict(input_n)[0]) + max(0, h_mods['gol'].predict(input_i)[0])) / 2
            xc_h = (max(0, h_mods['corn'].predict(input_n)[0]) + max(0, a_mods['corn'].predict(input_i)[0])) / 2
            xc_a = (max(0, a_mods['corn'].predict(input_n)[0]) + max(0, h_mods['corn'].predict(input_i)[0])) / 2
            xs_h = (max(0, h_mods['shot'].predict(input_n)[0]) + max(0, a_mods['shot'].predict(input_i)[0])) / 2
            xs_a = (max(0, a_mods['shot'].predict(input_n)[0]) + max(0, h_mods['shot'].predict(input_i)[0])) / 2
            xst_h = (max(0, h_mods['shot_t'].predict(input_n)[0]) + max(0, a_mods['shot_t'].predict(input_i)[0])) / 2
            xst_a = (max(0, a_mods['shot_t'].predict(input_n)[0]) + max(0, h_mods['shot_t'].predict(input_i)[0])) / 2
            xy_h = (max(0, h_mods['card'].predict(input_n)[0]) + max(0, a_mods['card'].predict(input_i)[0])) / 2
            xy_a = (max(0, a_mods['card'].predict(input_n)[0]) + max(0, h_mods['card'].predict(input_i)[0])) / 2
        else:
            p_h, p_d, p_a = pmap_n.get('H',0), pmap_n.get('D',0), pmap_n.get('A',0)
            xg_h = max(0, h_mods['gol'].predict(input_n)[0])
            xg_a = max(0, a_mods['gol'].predict(input_n)[0])
            xc_h = max(0, h_mods['corn'].predict(input_n)[0])
            xc_a = max(0, a_mods['corn'].predict(input_n)[0])
            xs_h = max(0, h_mods['shot'].predict(input_n)[0])
            xs_a = max(0, a_mods['shot'].predict(input_n)[0])
            xst_h = max(0, h_mods['shot_t'].predict(input_n)[0])
            xst_a = max(0, a_mods['shot_t'].predict(input_n)[0])
            xy_h = max(0, h_mods['card'].predict(input_n)[0])
            xy_a = max(0, a_mods['card'].predict(input_n)[0])
        
        tot_g = xg_h + xg_a; tot_c = xc_h + xc_a; tot_y = xy_h + xy_a
        
        # --- CÁLCULO DE MARCADOR PONDERADO SINCRONIZADO ---
        matriz_scores = []
        suma_probabilidades = 0
        for i in range(6): 
            for j in range(6): 
                prob_poisson = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a)
                if i > j:
                    peso_ml = p_h
                elif i < j:
                    peso_ml = p_a
                else:
                    peso_ml = p_d
                
                prob_combinada = prob_poisson * peso_ml
                suma_probabilidades += prob_combinada
                matriz_scores.append({'score': f"{i} - {j}", 'prob_combinada': prob_combinada})
                
        for score in matriz_scores:
            score['prob'] = (score['prob_combinada'] / suma_probabilidades) * 100 if suma_probabilidades > 0 else 0
            
        top_scores = sorted(matriz_scores, key=lambda x: x['prob'], reverse=True)[:3]

        def calc_poisson(expected, threshold): return poisson.sf(threshold, expected) * 100
        
        html = f"""
        <style>
            .card {{ font-family: 'Segoe UI', sans-serif; background: #fff; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border: 1px solid #ccc; overflow:hidden; width: 100%; }}
            .header {{ background: #0f172a; color: #fff; padding: 15px; text-align: center; font-weight: bold; font-size: 1em; border-bottom: 4px solid #10b981; }}
            .teams-row {{ display: flex; justify-content: space-around; align-items: center; padding: 15px; background: #f8fafc; flex-wrap: wrap; }}
            .team-nm {{ font-size: 1.2em; font-weight: bold; color: #1e293b; text-align: center; width: 40%; }}
            .vs-tag {{ background: #e2e8f0; color: #475569; padding: 5px 10px; border-radius: 4px; font-weight: 900; font-size: 0.8em; }}
            .win-bar {{ display: flex; height: 8px; margin: 0; }}
            .wb-part {{ height: 100%; }}
            .section-title {{ padding: 8px 10px; font-weight: bold; color: #f8fafc; background: #334155; font-size: 0.8em; text-transform: uppercase; }}
            .stats-table {{ width: 100%; text-align: center; border-collapse: collapse; }}
            .stats-table th {{ background: #f1f5f9; padding: 8px; font-size: 0.75em; color: #475569; border-bottom: 2px solid #e2e8f0; }}
            .stats-table td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: bold; font-size: 0.9em; }}
            .lbl-col {{ text-align: left !important; padding-left: 10px !important; color: #64748b !important; }}
            .flex-markets {{ display: flex; padding: 10px; gap: 10px; background: #f8fafc; flex-wrap: wrap; }}
            .mkt-box {{ flex: 1 1 45%; background: white; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; text-align: center; }}
            .mkt-title {{ font-size: 0.7em; color: #64748b; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; }}
            .mkt-val {{ font-size: 1.1em; font-weight: 900; color: #0f172a; }}
        </style>
        
        <div class="card">
            <div class="header">ANÁLISIS V15.1: {home[:15].upper()} VS {away[:15].upper()}</div>
            
            <div class="teams-row">
                <div class="team-nm">{home[:10]}</div>
                <div class="vs-tag">VS</div>
                <div class="team-nm">{away[:10]}</div>
            </div>
            
            <div class="win-bar">
                <div class="wb-part" style="width:{p_h*100}%; background:#3b82f6;"></div>
                <div class="wb-part" style="width:{p_d*100}%; background:#94a3b8;"></div>
                <div class="wb-part" style="width:{p_a*100}%; background:#ef4444;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; padding: 5px 10px; font-size:0.75em; font-weight:bold; background:white;">
                <span style="color:#3b82f6">{p_h*100:.1f}%</span>
                <span style="color:#64748b">EMP: {p_d*100:.1f}%</span>
                <span style="color:#ef4444">{p_a*100:.1f}%</span>
            </div>
            
            <div class="section-title">MÉTRICAS ESPERADAS (IA - RACHA 10)</div>
            <table class="stats-table">
                <tr>
                    <th class="lbl-col">MÉTRICA</th>
                    <th>{home[:3].upper()}</th>
                    <th>{away[:3].upper()}</th>
                    <th style="color:#166534">TOTAL</th>
                </tr>
                <tr><td class="lbl-col">⚽ xG</td><td>{xg_h:.2f}</td><td>{xg_a:.2f}</td><td style="color:#166534">{tot_g:.2f}</td></tr>
                <tr><td class="lbl-col">🚩 Córners</td><td>{xc_h:.1f}</td><td>{xc_a:.1f}</td><td style="color:#166534">{tot_c:.1f}</td></tr>
                <tr><td class="lbl-col">🎯 T. Arco</td><td style="color:#0284c7">{xst_h:.1f}</td><td style="color:#0284c7">{xst_a:.1f}</td><td style="color:#166534">{xst_h+xst_a:.1f}</td></tr>
                <tr><td class="lbl-col">🔫 T. Totales</td><td>{xs_h:.1f}</td><td>{xs_a:.1f}</td><td style="color:#166534">{xs_h+xs_a:.1f}</td></tr>
                <tr><td class="lbl-col">🟨 Tarjetas</td><td style="color:#d97706">{xy_h:.1f}</td><td style="color:#d97706">{xy_a:.1f}</td><td style="color:#166534">{tot_y:.1f}</td></tr>
            </table>

            <div class="section-title">MERCADOS ESTRATÉGICOS</div>
            <div class="flex-markets">
                <div class="mkt-box"><div class="mkt-title">Over 2.5</div><div class="mkt-val" style="color:{'#166534' if calc_poisson(tot_g, 2.5)>55 else '#991b1b'}">{calc_poisson(tot_g, 2.5):.1f}%</div></div>
                <div class="mkt-box"><div class="mkt-title">BTTS</div><div class="mkt-val">{(1-poisson.pmf(0, xg_h))*(1-poisson.pmf(0, xg_a))*100:.1f}%</div></div>
                <div class="mkt-box"><div class="mkt-title">Over 8.5 Córners</div><div class="mkt-val">{calc_poisson(tot_c, 8.5):.1f}%</div></div>
                <div class="mkt-box"><div class="mkt-title">Top Marcador</div><div class="mkt-val" style="color:#0284c7;">{top_scores[0]['score']} ({top_scores[0]['prob']:.0f}%)</div></div>
            </div>
        </div>
        """
        
        components.html(html, height=750, scrolling=True)
