import pandas as pd
import numpy as np

# ==============================================================================
# ⚙️ INGENIERÍA DE CARACTERÍSTICAS (EWMA & STATS)
# ==============================================================================

def calcular_ewma(valores, span=5):
    """
    Calcula la Media Móvil Exponencial (EWMA).
    Le da mayor peso a los partidos más recientes con un decaimiento suave.
    """
    if len(valores) == 0:
        return 0.0
    # Usamos Pandas para calcular el EWMA del historial
    serie = pd.Series(valores)
    return serie.ewm(span=span, adjust=False).mean().iloc[-1]

def preparar_features(df_historico):
    """
    Genera el dataset final (Features X y Targets Y) procesando el 
    historial cronológicamente para entrenar el modelo de Machine Learning.
    """
    # Identificamos a todos los equipos únicos
    equipos = pd.concat([df_historico['home_team'], df_historico['away_team']]).unique()
    
    # Diccionario para almacenar el historial progresivo de cada equipo
    team_stats = {eq: {'GF': [], 'GC': [], 'S': [], 'ST': [], 'C': [], 'Y': [], 'Pts': []} for eq in equipos}
    
    X_list = []
    targets = {'res':[], 'g_h':[], 'g_a':[], 'c_h':[], 'c_a':[], 's_h':[], 's_a':[], 'st_h':[], 'st_a':[], 't_h':[], 't_a':[]}
    
    # Recorremos el historial partido a partido
    for idx, row in df_historico.iterrows():
        h, a = row['home_team'], row['away_team']
        
        # 1. EXTRAEMOS LA INFORMACIÓN "ANTES" DEL PARTIDO (Previene Fuga de Datos)
        X_list.append({
            'H_GF': calcular_ewma(team_stats[h]['GF']),
            'H_GC': calcular_ewma(team_stats[h]['GC']),
            'H_S': calcular_ewma(team_stats[h]['S']),
            'H_ST': calcular_ewma(team_stats[h]['ST']),
            'H_C': calcular_ewma(team_stats[h]['C']),
            'H_Y': calcular_ewma(team_stats[h]['Y']),
            'H_Form': calcular_ewma(team_stats[h]['Pts']),
            
            'A_GF': calcular_ewma(team_stats[a]['GF']),
            'A_GC': calcular_ewma(team_stats[a]['GC']),
            'A_S': calcular_ewma(team_stats[a]['S']),
            'A_ST': calcular_ewma(team_stats[a]['ST']),
            'A_C': calcular_ewma(team_stats[a]['C']),
            'A_Y': calcular_ewma(team_stats[a]['Y']),
            'A_Form': calcular_ewma(team_stats[a]['Pts']),
            
            'Neutral': 1 if row.get('neutral', False) else 0,
            'Is_Qualifier': 1 if 'qualification' in str(row['tournament']).lower() else 0,
            
            # Integración del Sistema ELO (calculado en el Paso 2)
            'H_Elo': row['Home_Elo'],
            'A_Elo': row['Away_Elo'],
            'Elo_Diff': row['Elo_Diff']
        })
        
        # 2. DEFINIMOS LOS OBJETIVOS (Lo que la IA aprenderá a predecir)
        if row['home_score'] > row['away_score']: ftr = 'H'
        elif row['home_score'] < row['away_score']: ftr = 'A'
        else: ftr = 'D'
        
        targets['res'].append(ftr)
        targets['g_h'].append(row['home_score']); targets['g_a'].append(row['away_score'])
        targets['c_h'].append(row['HC']); targets['c_a'].append(row['AC'])
        targets['s_h'].append(row['HS']); targets['s_a'].append(row['AS'])
        targets['st_h'].append(row['HST']); targets['st_a'].append(row['AST'])
        targets['t_h'].append(row['HY']); targets['t_a'].append(row['AY'])
        
        # 3. ACTUALIZAMOS LOS HISTORIALES "DESPUÉS" DEL PARTIDO
        pts_h = 3 if ftr == 'H' else (1 if ftr == 'D' else 0)
        pts_a = 3 if ftr == 'A' else (1 if ftr == 'D' else 0)
        
        # Estadísticas del equipo local
        team_stats[h]['Pts'].append(pts_h)
        team_stats[h]['GF'].append(row['home_score'])
        team_stats[h]['GC'].append(row['away_score'])
        team_stats[h]['S'].append(row['HS'])
        team_stats[h]['ST'].append(row['HST'])
        team_stats[h]['C'].append(row['HC'])
        team_stats[h]['Y'].append(row['HY'])

        # Estadísticas del equipo visitante
        team_stats[a]['Pts'].append(pts_a)
        team_stats[a]['GF'].append(row['away_score'])
        team_stats[a]['GC'].append(row['home_score'])
        team_stats[a]['S'].append(row['AS'])
        team_stats[a]['ST'].append(row['AST'])
        team_stats[a]['C'].append(row['AC'])
        team_stats[a]['Y'].append(row['AY'])
        
    X = pd.DataFrame(X_list).fillna(0)
    return X, targets, team_stats
