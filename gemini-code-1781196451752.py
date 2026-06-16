# ==============================================================================
# 📱 ULTIMATE AI V15.3 - MONTECARLO & TIME DECAY (UI PREMIUM GERENCIAL)
# ==============================================================================

import streamlit as st
import streamlit.components.v1 as components  
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from collections import Counter
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
            arr = team_stats[t][stat][-12:]
            if len(arr) == 0: return 0.0
            pesos = np.linspace(0.5, 1.5, len(arr))
            return np.average(arr, weights=pesos)
            
        def get_form(t): 
            arr = team_stats[t]['Pts'][-12:]
            if len(arr) == 0: return 0.0
            pesos = np.linspace(0.5, 1.5, len(arr))
            return np.sum(arr * pesos)
            
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

st.title("🏆 ULTIMATE AI V15.5")
st.markdown("### Predicciones IA (Montecarlo & UI Premium)")

df_global = cargar_y_enriquecer_selecciones()

# Escudo anti-errores: Si el archivo no existe o tiene menos de 2 partidos, se detiene amablemente
if df_global is None or len(df_global) < 2:
    st.warning("⚠️ Base de datos vacía. Por favor, asegúrate de que el archivo 'datos_reales_selecciones.csv' contenga partidos para que la IA pueda entrenar.")
    st.stop()

with st.spinner('Procesando Algoritmos Analíticos...'):
    clf, le, h_mods, a_mods, stats = entrenar_ia(df_global)

equipos_activos = sorted([eq for eq in stats.keys() if len(stats[eq]['Pts']) > 0])

col1, col2 = st.columns(2)
with col1: home = st.selectbox("🌍 Equipo 1 (Local):", equipos_activos, index=0)
with col2: away = st.selectbox("🌍 Equipo 2 (Visita):", equipos_activos, index=1 if len(equipos_activos)>1 else 0)

col_opt1, col_opt2 = st.columns(2)
with col_opt1: is_neutral = st.checkbox("Cancha Neutral", value=True)
with col_opt2: is_qualifier = st.checkbox("Partido Oficial", value=True)

if st.button("🚀 GENERAR INFORME DIRECTIVO", use_container_width=True):
    if home == away:
        st.error("⚠️ Error: Selecciona equipos distintos.")
    else:
        def get_avg_predict(t_data, stat):
            arr = t_data[stat][-12:]
            if len(arr) == 0: return 0.0
            pesos = np.linspace(0.5, 1.5, len(arr))
            return np.average(arr, weights=pesos)
            
        def get_form_predict(t_data): 
            arr = t_data['Pts'][-12:]
            if len(arr) == 0: return 0.0
            pesos = np.linspace(0.5, 1.5, len(arr))
            return np.sum(arr * pesos)
            
        fecha_actual = pd.Timestamp.now()
        
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

        input_n = generar_input_ia(home, away)
        probs_n = clf.predict_proba(input_n)[0]
        cls = le.inverse_transform(clf.classes_)
        pmap_n = {c: p for c, p in zip(cls, probs_n)}
        
        if is_neutral:
            input_i = generar_input_ia(away, home)
            probs_i = clf.predict_proba(input_i)[0]
            pmap_i = {c: p for c, p in zip(cls, probs_i)}
            
            p_h = (pmap_n.get('H', 0) + pmap_i.get('A', 0)) / 2
            p_a = (pmap_n.get('A', 0) + pmap_i.get('H', 0)) / 2
            p_d = (pmap_n.get('D', 0) + pmap_i.get('D', 0)) / 2
            
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
        
        # Simulación de Montecarlo
        iteraciones = 100000
        sim_h = np.random.poisson(xg_h, iteraciones)
        sim_a = np.random.poisson(xg_a, iteraciones)
        
        resultados_simulados = []
        for i in range(iteraciones):
            h_goles = sim_h[i]
            a_goles = sim_a[i]
            peso_ml = p_h if h_goles > a_goles else (p_a if h_goles < a_goles else p_d)
            resultados_simulados.append({'score': f"{h_goles} - {a_goles}", 'peso': peso_ml})
        <div class="header">REPORTE DIRECTIVO V15.5: {home[:15].upper()} VS {away[:15].upper()}</div>   
        df_sim = pd.DataFrame(resultados_simulados)
        df_agrupado = df_sim.groupby('score')['peso'].sum().reset_index()
        total_peso = df_agrupado['peso'].sum()
        df_agrupado['prob'] = (df_agrupado['peso'] / total_peso) * 100
        top_scores = df_agrupado.sort_values(by='prob', ascending=False).head(5).to_dict('records')
        
        # --- CÁLCULO DE PROBABILIDADES ESTADÍSTICAS ---
        def calc_poisson(expected, threshold): return poisson.sf(threshold, expected) * 100
        
        prob_over25 = calc_poisson(tot_g, 2.5)
        prob_btts = (1 - poisson.pmf(0, xg_h)) * (1 - poisson.pmf(0, xg_a)) * 100
        
        # --- LÓGICA DEL SEMÁFORO DE RIESGO ---
        def get_color_riesgo(probabilidad):
            if probabilidad >= 60: return "#166534" # Verde (Bajo Riesgo / Oportunidad)
            elif probabilidad >= 45: return "#d97706" # Naranja (Riesgo Medio)
            else: return "#991b1b" # Rojo (Alto Riesgo)

        # 👇 AGREGA ESTAS DOS LÍNEAS QUE FALTABAN 👇
        color_o25 = get_color_riesgo(prob_over25)
        color_btts = get_color_riesgo(prob_btts)
        # 👆 ------------------------------------ 👆

        # --- BARRAS DE CALOR PARA MONTECARLO ---
        filas_tabla_marcadores = ""
        max_prob = top_scores[0]['prob'] if top_scores else 100
        for s in top_scores:
            ancho_barra = (s['prob'] / max_prob) * 100 if max_prob > 0 else 0
            barra_html = f"""
            <div style='background: #e2e8f0; border-radius: 4px; position: relative; width: 100%; height: 24px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);'>
                <div style='background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%); width: {ancho_barra}%; height: 100%; position: absolute; top: 0; left: 0; border-radius: 4px;'></div>
                <span style='position: relative; z-index: 1; padding-left: 10px; font-weight: 800; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.6); line-height: 24px; font-size: 0.9em;'>{s['prob']:.1f}%</span>
            </div>
            """
            filas_tabla_marcadores += f"<tr class='hover-row'><td style='font-size:1.1em; font-weight: 900; color:#1e293b;'>{s['score']}</td><td style='padding: 8px 10px;'>{barra_html}</td></tr>"

        html = f"""
        <style>
            .card {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #fff; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; overflow:hidden; width: 100%; margin-bottom: 20px; }}
            .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #fff; padding: 16px; text-align: center; font-weight: 800; font-size: 1.1em; border-bottom: 4px solid #10b981; letter-spacing: 1px; }}
            .teams-row {{ display: flex; justify-content: space-around; align-items: center; padding: 20px; background: #f8fafc; flex-wrap: wrap; border-bottom: 1px solid #e2e8f0; }}
            .team-nm {{ font-size: 1.3em; font-weight: 900; color: #0f172a; text-align: center; width: 40%; text-transform: uppercase; }}
            .vs-tag {{ background: #cbd5e1; color: #334155; padding: 6px 12px; border-radius: 8px; font-weight: 900; font-size: 0.85em; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }}
            .win-bar {{ display: flex; height: 10px; margin: 0; }}
            .wb-part {{ height: 100%; transition: width 0.5s ease; }}
            .section-title {{ padding: 10px 15px; font-weight: 800; color: #f8fafc; background: #334155; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }}
            .stats-table {{ width: 100%; text-align: center; border-collapse: collapse; }}
            .stats-table th {{ background: #f1f5f9; padding: 10px; font-size: 0.8em; color: #475569; border-bottom: 2px solid #cbd5e1; text-transform: uppercase; }}
            .stats-table td {{ padding: 12px 10px; border-bottom: 1px solid #e2e8f0; font-weight: 700; font-size: 0.95em; color: #334155; }}
            .stats-table tr.zebra:nth-child(even) {{ background-color: #f8fafc; }}
            .stats-table tr.hover-row:hover {{ background-color: #f1f5f9; transition: background 0.2s; }}
            .lbl-col {{ text-align: left !important; padding-left: 15px !important; color: #64748b !important; border-right: 1px solid #e2e8f0; cursor: help; }}
            .flex-markets {{ display: flex; padding: 15px; gap: 15px; background: #f8fafc; flex-wrap: wrap; }}
            .mkt-box {{ flex: 1 1 45%; background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s, box-shadow 0.2s; }}
            .mkt-box:hover {{ transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }}
            .mkt-title {{ font-size: 0.75em; color: #64748b; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }}
            .mkt-val {{ font-size: 1.4em; font-weight: 900; }}
        </style>
        
        <div class="card">
            <div class="header">REPORTE DIRECTIVO V15.3: {home[:15].upper()} VS {away[:15].upper()}</div>
            
            <div class="teams-row">
                <div class="team-nm">{home[:10]}</div>
                <div class="vs-tag">VS</div>
                <div class="team-nm">{away[:10]}</div>
            </div>
            
            <div class="win-bar">
                <div class="wb-part" style="width:{p_h*100}%; background:#3b82f6;" title="Probabilidad Local"></div>
                <div class="wb-part" style="width:{p_d*100}%; background:#94a3b8;" title="Probabilidad Empate"></div>
                <div class="wb-part" style="width:{p_a*100}%; background:#ef4444;" title="Probabilidad Visita"></div>
            </div>
            <div style="display:flex; justify-content:space-between; padding: 8px 15px; font-size:0.8em; font-weight:800; background:white; border-bottom: 1px solid #e2e8f0;">
                <span style="color:#3b82f6">{p_h*100:.1f}%</span>
                <span style="color:#64748b">EMP: {p_d*100:.1f}%</span>
                <span style="color:#ef4444">{p_a*100:.1f}%</span>
            </div>
            
            <div class="section-title">MÉTRICAS ESPERADAS (TIME DECAY - ÚLT. 12)</div>
            <table class="stats-table">
                <tr>
                    <th class="lbl-col">MÉTRICA</th>
                    <th>{home[:3].upper()}</th>
                    <th>{away[:3].upper()}</th>
                    <th style="color:#0f172a">TOTAL</th>
                </tr>
                <tr class="zebra hover-row"><td class="lbl-col" title="Goles Esperados (xG): Calidad matemática de las ocasiones de gol generadas.">⚽ xG</td><td>{xg_h:.2f}</td><td>{xg_a:.2f}</td><td style="color:#0f172a">{tot_g:.2f}</td></tr>
                <tr class="zebra hover-row"><td class="lbl-col" title="Tiros de esquina proyectados según el volumen ofensivo.">🚩 Córners</td><td>{xc_h:.1f}</td><td>{xc_a:.1f}</td><td style="color:#0f172a">{tot_c:.1f}</td></tr>
                <tr class="zebra hover-row"><td class="lbl-col" title="Tiros al Arco: Disparos que van directamente entre los tres palos.">🎯 T. Arco</td><td style="color:#0284c7">{xst_h:.1f}</td><td style="color:#0284c7">{xst_a:.1f}</td><td style="color:#0f172a">{xst_h+xst_a:.1f}</td></tr>
                <tr class="zebra hover-row"><td class="lbl-col" title="Tiros Totales: Suma de todos los remates (desviados, bloqueados y a puerta).">🔫 T. Totales</td><td>{xs_h:.1f}</td><td>{xs_a:.1f}</td><td style="color:#0f172a">{xs_h+xs_a:.1f}</td></tr>
                <tr class="zebra hover-row"><td class="lbl-col" title="Proyección de tarjetas (Fricción del partido).">🟨 Tarjetas</td><td style="color:#d97706">{xy_h:.1f}</td><td style="color:#d97706">{xy_a:.1f}</td><td style="color:#0f172a">{tot_y:.1f}</td></tr>
            </table>

            <div class="section-title">SEMÁFORO DE RIESGO ESTRATÉGICO</div>
            <div class="flex-markets">
                <div class="mkt-box">
                    <div class="mkt-title" title="Probabilidad de que el partido termine con 3 goles o más.">Over 2.5 Goles</div>
                    <div class="mkt-val" style="color: {color_o25};">{prob_over25:.1f}%</div>
                </div>
                <div class="mkt-box">
                    <div class="mkt-title" title="Ambos Equipos Anotan (Sí).">BTTS - Sí</div>
                    <div class="mkt-val" style="color: {color_btts};">{prob_btts:.1f}%</div>
                </div>
            </div>
            
            <div class="section-title" style="background: linear-gradient(90deg, #0f172a 0%, #334155 100%);">🎲 TOP 5 MARCADORES (100,000 SIMULACIONES)</div>
            <table class="stats-table">
                <tr>
                    <th style="width: 30%;">MARCADOR</th>
                    <th style="text-align: left; padding-left: 10px;">PROBABILIDAD & DISTRIBUCIÓN</th>
                </tr>
                {filas_tabla_marcadores}
            </table>
        </div>
        """
        
        components.html(html, height=950, scrolling=True)
