import streamlit as st
import streamlit.components.v1 as components  
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import os

# Importamos nuestros módulos del Engine
from engine.elo_system import obtener_datos_con_elo
from engine.features import preparar_features, calcular_ewma

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Predicciones", page_icon="🏆", layout="centered")

@st.cache_resource(show_spinner=False)
def entrenar_ia():
    """Carga datos, procesa features y entrena los modelos de Machine Learning"""
    df, elo_dict = obtener_datos_con_elo('data/futbol.db')
    
    if df is None or len(df) < 10:
        return None, None, None, None, None, None
        
    X, targets, team_stats = preparar_features(df)
    
    # Entrenamos el Clasificador Principal (1 X 2)
    le = LabelEncoder()
    y_res = le.fit_transform(targets['res']) 
    rs = 42 
    
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=rs)
    clf.fit(X, y_res)
    
    # Entrenamos los Regresores para el Equipo Local
    h_mods = {
        'gol': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['g_h']),
        'corn': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['c_h']),
        'shot': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['s_h']),
        'shot_t': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['st_h']),
        'card': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['t_h'])
    }
    
    # Entrenamos los Regresores para el Equipo Visitante
    a_mods = {
        'gol': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['g_a']),
        'corn': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['c_a']),
        'shot': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['s_a']),     
        'shot_t': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['st_a']), 
        'card': GradientBoostingRegressor(n_estimators=50, random_state=rs).fit(X, targets['t_a'])     
    }
    
    return clf, le, h_mods, a_mods, team_stats, elo_dict

# ==============================================================================
# UI PRINCIPAL DE STREAMLIT
# ==============================================================================

st.title("🏆 ULTIMATE AI")
st.markdown("### Motor Predictivo Modular (Elo + EWMA)")

with st.spinner('Inicializando Base de Datos y Entrenando Modelos...'):
    clf, le, h_mods, a_mods, team_stats, elos = entrenar_ia()

if clf is None:
    st.warning("⚠️ Base de datos vacía o insuficiente. Ejecuta 'python engine/data_bot.py' primero.")
    st.stop()

equipos_activos = sorted([eq for eq, stats in team_stats.items() if len(stats['Pts']) > 0])

col1, col2 = st.columns(2)
with col1: home = st.selectbox("🌍 Equipo 1 (Local):", equipos_activos, index=0)
with col2: away = st.selectbox("🌍 Equipo 2 (Visita):", equipos_activos, index=1 if len(equipos_activos)>1 else 0)

col_opt1, col_opt2 = st.columns(2)
with col_opt1: is_neutral = st.checkbox("Cancha Neutral", value=True)
with col_opt2: is_qualifier = st.checkbox("Partido Oficial", value=True)

if st.button("🚀 GENERAR INFORME GERENCIAL", use_container_width=True):
    if home == away:
        st.error("⚠️ Error: Selecciona equipos distintos.")
    else:
        # Generar el input exacto usando el estado de forma actual y el Elo
        def generar_input_ia(t_local, t_visita):
            h_data, a_data = team_stats[t_local], team_stats[t_visita]
            return pd.DataFrame([{
                'H_GF': calcular_ewma(h_data['GF']), 'H_GC': calcular_ewma(h_data['GC']),
                'H_S': calcular_ewma(h_data['S']), 'H_ST': calcular_ewma(h_data['ST']),
                'H_C': calcular_ewma(h_data['C']), 'H_Y': calcular_ewma(h_data['Y']),
                'H_Form': calcular_ewma(h_data['Pts']),
                'A_GF': calcular_ewma(a_data['GF']), 'A_GC': calcular_ewma(a_data['GC']),
                'A_S': calcular_ewma(a_data['S']), 'A_ST': calcular_ewma(a_data['ST']),
                'A_C': calcular_ewma(a_data['C']), 'A_Y': calcular_ewma(a_data['Y']),
                'A_Form': calcular_ewma(a_data['Pts']),
                'Neutral': 1 if is_neutral else 0,
                'Is_Qualifier': 1 if is_qualifier else 0,
                'H_Elo': elos.get(t_local, 1500),
                'A_Elo': elos.get(t_visita, 1500),
                'Elo_Diff': elos.get(t_local, 1500) - elos.get(t_visita, 1500)
            }])

        input_n = generar_input_ia(home, away)
        probs_n = clf.predict_proba(input_n)[0]
        cls = le.inverse_transform(clf.classes_)
        pmap_n = {c: p for c, p in zip(cls, probs_n)}
        
        # Ponderación para cancha neutral (promedia ida y vuelta simulada)
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
        else:
            p_h, p_d, p_a = pmap_n.get('H',0), pmap_n.get('D',0), pmap_n.get('A',0)
            xg_h = max(0, h_mods['gol'].predict(input_n)[0])
            xg_a = max(0, a_mods['gol'].predict(input_n)[0])
            xc_h = max(0, h_mods['corn'].predict(input_n)[0])
            xc_a = max(0, a_mods['corn'].predict(input_n)[0])
        
        tot_g = xg_h + xg_a; tot_c = xc_h + xc_a
        
        # Simulación Montecarlo
        iteraciones = 100000
        sim_h = np.random.poisson(xg_h, iteraciones)
        sim_a = np.random.poisson(xg_a, iteraciones)
        
        resultados_simulados = []
        for i in range(iteraciones):
            peso_ml = p_h if sim_h[i] > sim_a[i] else (p_a if sim_h[i] < sim_a[i] else p_d)
            resultados_simulados.append({'score': f"{sim_h[i]} - {sim_a[i]}", 'peso': peso_ml})
            
        df_sim = pd.DataFrame(resultados_simulados)
        df_agrupado = df_sim.groupby('score')['peso'].sum().reset_index()
        df_agrupado['prob'] = (df_agrupado['peso'] / df_agrupado['peso'].sum()) * 100
        top_scores = df_agrupado.sort_values(by='prob', ascending=False).head(5).to_dict('records')
        
        prob_over25 = poisson.sf(2.5, tot_g) * 100
        prob_btts = (1 - poisson.pmf(0, xg_h)) * (1 - poisson.pmf(0, xg_a)) * 100
        
        # Generación del formato de Informe Gerencial resumido
        favorito = home if p_h > p_a else away
        prob_fav = max(p_h, p_a) * 100
        elo_diff = abs(elos.get(home, 1500) - elos.get(away, 1500))
        
        texto_gerencial = f"""
        <strong>Resumen Ejecutivo:</strong> El análisis predictivo otorga un {prob_fav:.1f}% de probabilidad de victoria a {favorito}, 
        respaldado por una diferencia de {elo_diff:.0f} puntos en el rating Elo de la IA. El encuentro proyecta una 
        tendencia ofensiva de {tot_g:.2f} goles esperados (xG), con un {prob_over25:.1f}% de probabilidades de superar 
        la línea de 2.5 goles. El marcador exacto más probable de acuerdo a las simulaciones de Montecarlo 
        es el {top_scores[0]['score']} ({top_scores[0]['prob']:.1f}%).
        """

        # Construcción HTML
        filas_tabla = ""
        max_prob = top_scores[0]['prob'] if top_scores else 100
        for s in top_scores:
            ancho_barra = (s['prob'] / max_prob) * 100 if max_prob > 0 else 0
            filas_tabla += f"<tr><td style='font-size:1.1em; font-weight: 900; color:#1e293b;'>{s['score']}</td><td style='padding: 8px 10px;'><div style='background: #e2e8f0; border-radius: 4px; position: relative; width: 100%; height: 24px;'><div style='background: linear-gradient(90deg, #38bdf8 0%, #0284c7 100%); width: {ancho_barra}%; height: 100%; border-radius: 4px;'></div><span style='position: absolute; left: 10px; top:0; font-weight: 800; color: #fff; text-shadow: 1px 1px 2px rgba(0,0,0,0.6); line-height: 24px; font-size: 0.9em;'>{s['prob']:.1f}%</span></div></td></tr>"

        html = f"""
        <style>
            .card {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #fff; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; overflow:hidden; width: 100%; }}
            .header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #fff; padding: 16px; text-align: center; font-weight: 800; border-bottom: 4px solid #10b981; }}
            .paragraph-summary {{ padding: 20px; background-color: #f8fafc; color: #334155; font-size: 0.95em; line-height: 1.6; border-bottom: 1px solid #e2e8f0; text-align: justify; }}
            .teams-row {{ display: flex; justify-content: space-around; align-items: center; padding: 15px; border-bottom: 1px solid #e2e8f0; }}
            .team-nm {{ font-size: 1.2em; font-weight: 900; color: #0f172a; text-transform: uppercase; }}
            .stats-table {{ width: 100%; text-align: center; border-collapse: collapse; }}
            .stats-table th {{ background: #f1f5f9; padding: 10px; font-size: 0.8em; color: #475569; }}
            .stats-table td {{ padding: 12px 10px; border-bottom: 1px solid #e2e8f0; font-weight: 700; color: #334155; }}
        </style>
        <div class="card">
            <div class="header">INFORME GERENCIAL V16.0</div>
            <div class="paragraph-summary">
                {texto_gerencial}
            </div>
            <div class="teams-row">
                <div class="team-nm">{home[:12]}</div>
                <div style="font-weight:900; color:#64748b;">VS</div>
                <div class="team-nm">{away[:12]}</div>
            </div>
            <div style="display:flex; height: 12px; width: 100%;">
                <div style="width:{p_h*100}%; background:#3b82f6;"></div>
                <div style="width:{p_d*100}%; background:#94a3b8;"></div>
                <div style="width:{p_a*100}%; background:#ef4444;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; padding: 8px 15px; font-size:0.8em; font-weight:800; border-bottom: 1px solid #e2e8f0;">
                <span style="color:#3b82f6">{p_h*100:.1f}%</span><span style="color:#64748b">EMP: {p_d*100:.1f}%</span><span style="color:#ef4444">{p_a*100:.1f}%</span>
            </div>
            <table class="stats-table">
                <tr><th style="text-align: left; padding-left: 15px;">MÉTRICA</th><th>{home[:3].upper()}</th><th>{away[:3].upper()}</th><th>TOTAL</th></tr>
                <tr><td style="text-align: left; padding-left: 15px;">⚽ xG Esperados</td><td>{xg_h:.2f}</td><td>{xg_a:.2f}</td><td>{tot_g:.2f}</td></tr>
                <tr style="background:#f8fafc;"><td style="text-align: left; padding-left: 15px;">🚩 Córners</td><td>{xc_h:.1f}</td><td>{xc_a:.1f}</td><td>{tot_c:.1f}</td></tr>
            </table>
            <div style="padding: 10px 15px; font-weight: 800; color: #fff; background: #334155; font-size: 0.85em;">🎲 SIMULACIÓN MONTECARLO (100k iteraciones)</div>
            <table class="stats-table">
                {filas_tabla}
            </table>
        </div>
        """
        components.html(html, height=750, scrolling=True)
