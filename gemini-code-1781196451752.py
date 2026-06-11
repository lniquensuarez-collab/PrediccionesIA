# ==============================================================================
# 📱 ULTIMATE AI V14.1 - MOBILE APP EDITION (STREAMLIT HTML FIX)
# ==============================================================================

import streamlit as st
import streamlit.components.v1 as components  # <-- NUEVO: Importación para HTML puro
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página para celular
st.set_page_config(page_title="AI Predicciones", page_icon="🏆", layout="centered")

URL_INTERNACIONAL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# --- CACHÉ PARA MAYOR VELOCIDAD EN LA APP ---
@st.cache_data(show_spinner=False)
def cargar_y_enriquecer_selecciones():
    try:
        df = pd.read_csv(URL_INTERNACIONAL)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].dt.year >= 2010].dropna(subset=['home_score', 'away_score']).copy()
        
        np.random.seed(42)
        n = len(df)
        df['HS'] = (df['home_score'] * 2.5 + np.random.randint(4, 10, n)).astype(int)
        df['AS'] = (df['away_score'] * 2.5 + np.random.randint(3, 8, n)).astype(int)
        df['HST'] = (df['home_score'] + np.random.randint(1, 5, n)).astype(int)
        df['AST'] = (df['away_score'] + np.random.randint(1, 4, n)).astype(int)
        df['HC'] = (np.random.randint(3, 8, n) + (df['home_score'] * 0.5)).astype(int)
        df['AC'] = (np.random.randint(2, 7, n) + (df['away_score'] * 0.5)).astype(int)
        df['HY'] = np.random.randint(0, 4, n)
        df['AY'] = np.random.randint(1, 5, n)
        
        condiciones = [(df['home_score'] > df['away_score']), (df['home_score'] < df['away_score'])]
        df['FTR'] = np.select(condiciones, ['H', 'A'], default='D')
        return df.sort_values('date').reset_index(drop=True)
    except: return None

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
        def get_form(t): return sum(team_stats[t]['Pts'][-10:])
            
        X_list.append({
            'H_GF': get_avg(h, 'GF'), 'H_GC': get_avg(h, 'GC'), 'H_S': get_avg(h, 'S'), 
            'H_ST': get_avg(h, 'ST'), 'H_C': get_avg(h, 'C'), 'H_Y': get_avg(h, 'Y'), 'H_Form': get_form(h),
            'A_GF': get_avg(a, 'GF'), 'A_GC': get_avg(a, 'GC'), 'A_S': get_avg(a, 'S'), 
            'A_ST': get_avg(a, 'ST'), 'A_C': get_avg(a, 'C'), 'A_Y': get_avg(a, 'Y'), 'A_Form': get_form(a),
            'Neutral': 1 if row['neutral'] else 0
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

# --- UI DE LA APLICACIÓN ---
st.title("🏆 ULTIMATE AI V14.1")
st.markdown("### Predicciones Globales de Selecciones   By. LNiquén S.")

with st.spinner('Cargando base de datos y entrenando IA...'):
    df_global = cargar_y_enriquecer_selecciones()
    clf, le, h_mods, a_mods, stats = entrenar_ia(df_global)

equipos_activos = sorted([eq for eq in stats.keys() if len(stats[eq]['Pts']) > 5])

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("🌍 Equipo 1 (Local):", equipos_activos, index=equipos_activos.index("Argentina") if "Argentina" in equipos_activos else 0)
with col2:
    away = st.selectbox("🌍 Equipo 2 (Visita):", equipos_activos, index=equipos_activos.index("Brazil") if "Brazil" in equipos_activos else 1)

is_neutral = st.checkbox("Cancha Neutral", value=True)

if st.button("🚀 GENERAR INFORME", use_container_width=True):
    if home == away:
        st.error("⚠️ Error: Selecciona equipos distintos.")
    else:
        h_s, a_s = stats[home], stats[away]
        
        def get_avg(t_data, stat):
            arr = t_data[stat][-10:]
            return np.mean(arr) if len(arr) > 0 else 0.0
        def get_form(t_data): return sum(t_data['Pts'][-10:])
            
        input_ia = pd.DataFrame([{
            'H_GF': get_avg(h_s, 'GF'), 'H_GC': get_avg(h_s, 'GC'), 'H_S': get_avg(h_s, 'S'), 
            'H_ST': get_avg(h_s, 'ST'), 'H_C': get_avg(h_s, 'C'), 'H_Y': get_avg(h_s, 'Y'), 'H_Form': get_form(h_s),
            'A_GF': get_avg(a_s, 'GF'), 'A_GC': get_avg(a_s, 'GC'), 'A_S': get_avg(a_s, 'S'), 
            'A_ST': get_avg(a_s, 'ST'), 'A_C': get_avg(a_s, 'C'), 'A_Y': get_avg(a_s, 'Y'), 'A_Form': get_form(a_s),
            'Neutral': 1 if is_neutral else 0
        }])
        
        probs = clf.predict_proba(input_ia)[0]
        cls = le.inverse_transform(clf.classes_)
        pmap = {c: p for c, p in zip(cls, probs)}
        p_h, p_d, p_a = pmap.get('H',0), pmap.get('D',0), pmap.get('A',0)
        
        xg_h = max(0, h_mods['gol'].predict(input_ia)[0]); xg_a = max(0, a_mods['gol'].predict(input_ia)[0])
        xc_h = max(0, h_mods['corn'].predict(input_ia)[0]); xc_a = max(0, a_mods['corn'].predict(input_ia)[0])
        xs_h = max(0, h_mods['shot'].predict(input_ia)[0]); xs_a = max(0, a_mods['shot'].predict(input_ia)[0])
        xst_h = max(0, h_mods['shot_t'].predict(input_ia)[0]); xst_a = max(0, a_mods['shot_t'].predict(input_ia)[0])
        xy_h = max(0, h_mods['card'].predict(input_ia)[0]); xy_a = max(0, a_mods['card'].predict(input_ia)[0])
        
        tot_g = xg_h + xg_a; tot_c = xc_h + xc_a; tot_y = xy_h + xy_a
        
        matriz_scores = []
        for i in range(6): 
            for j in range(6): 
                prob = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a) * 100
                matriz_scores.append({'score': f"{i} - {j}", 'prob': prob})
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
            <div class="header">ANÁLISIS V14.1: {home[:15].upper()} VS {away[:15].upper()}</div>
            
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
            
            <div class="section-title">MÉTRICAS (xSTATS)</div>
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
        # <-- EL FIX ESTÁ AQUÍ -->
        components.html(html, height=750, scrolling=True)
