import pandas as pd
import sqlite3

# ==============================================================================
# 🧠 SISTEMA ELO - CÁLCULO DE FUERZA RELATIVA
# ==============================================================================

# Parámetros iniciales
ELO_BASE = 1500  # Todos los equipos empiezan con 1500 puntos
K_FACTOR = 30    # Define qué tan rápido cambia el Elo. Mayor K = más volatilidad

def probabilidad_esperada(rating_a, rating_b):
    """Calcula la probabilidad (0 a 1) de ganar del equipo A contra el equipo B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def actualizar_elo(rating_actual, score_esperado, score_real, k_factor):
    """
    Actualiza el puntaje Elo basado en el resultado.
    score_real: 1 (Gana), 0.5 (Empata), 0 (Pierde)
    """
    return rating_actual + k_factor * (score_real - score_esperado)

def procesar_elo_historico(df):
    """
    Recorre el DataFrame de partidos cronológicamente.
    Calcula el Elo de cada equipo ANTES del partido para entrenar a la IA sin hacer trampa.
    Devuelve el DataFrame con el historial y un diccionario con los Elos actualizados a hoy.
    """
    # Nos aseguramos de que los partidos estén en orden cronológico
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    elo_dict = {}
    home_elos = []
    away_elos = []

    for index, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Si el equipo es nuevo en el historial, lo iniciamos con el Elo base
        if home not in elo_dict: elo_dict[home] = ELO_BASE
        if away not in elo_dict: elo_dict[away] = ELO_BASE
        
        # 1. Guardamos el Elo actual (ANTES del partido)
        elo_home_antes = elo_dict[home]
        elo_away_antes = elo_dict[away]
        
        home_elos.append(elo_home_antes)
        away_elos.append(elo_away_antes)
        
        # 2. Determinamos el resultado real del partido
        if row['home_score'] > row['away_score']:
            score_home, score_away = 1.0, 0.0
        elif row['home_score'] < row['away_score']:
            score_home, score_away = 0.0, 1.0
        else:
            score_home, score_away = 0.5, 0.5
            
        # 3. Calculamos la probabilidad esperada de que cada uno gane
        prob_home = probabilidad_esperada(elo_home_antes, elo_away_antes)
        prob_away = probabilidad_esperada(elo_away_antes, elo_home_antes)
        
        # MODIFICADOR ESTRATÉGICO: Los amistosos cambian menos el Elo que un Mundial
        if row['tournament'] == 'Friendlies':
            k_actual = 15  # Menor impacto
        else:
            k_actual = K_FACTOR # Impacto normal/alto para partidos oficiales
            
        # 4. Actualizamos los Elos en el diccionario (DESPUÉS del partido)
        elo_dict[home] = actualizar_elo(elo_home_antes, prob_home, score_home, k_actual)
        elo_dict[away] = actualizar_elo(elo_away_antes, prob_away, score_away, k_actual)

    # Añadimos las nuevas métricas al DataFrame
    df['Home_Elo'] = home_elos
    df['Away_Elo'] = away_elos
    # Esta diferencia es una variable súper predictiva para tu modelo de IA:
    df['Elo_Diff'] = df['Home_Elo'] - df['Away_Elo'] 
    
    return df, elo_dict

def obtener_datos_con_elo(db_path='data/futbol.db'):
    """Función principal que conectará la base de datos con el cálculo matemático."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM partidos", conn)
        conn.close()
        
        # Llamamos al procesador
        df_historico_elo, elo_actual_dict = procesar_elo_historico(df)
        return df_historico_elo, elo_actual_dict
    except Exception as e:
        print(f"Error cargando la base de datos: {e}")
        return None, None

# Pequeño test para comprobar que funciona (solo se ejecuta si corres este archivo directamente)
if __name__ == "__main__":
    df, elos = obtener_datos_con_elo('../data/futbol.db')
    if df is not None:
        print("✅ Sistema Elo procesado con éxito.")
        print(f"Total de partidos analizados: {len(df)}")
        # Mostrar el TOP 5 de selecciones con mejor Elo actual
        top_5 = sorted(elos.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n🏆 TOP 5 SELECCIONES ACTUAL (RANKING ELO IA):")
        for i, (equipo, puntaje) in enumerate(top_5, 1):
            print(f"{i}. {equipo}: {puntaje:.0f} puntos")
