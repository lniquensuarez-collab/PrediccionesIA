import pandas as pd
import requests
import time
import os
import sqlite3

# ==============================================================================
# 🤖 BOT EXTRACTOR V4.0 - MIGRACIÓN A SQLITE
# ==============================================================================

API_KEY = os.environ.get("RAPIDAPI_KEY") 
HEADERS = {
    "x-apisports-key": API_KEY
}

COMPETICIONES = {
    "World Cup": 1, "Euro": 4, "Copa America": 9, "Africa Cup of Nations": 6,
    "Copa Asiática": 807, "Copa Oro": 22, "UEFA Nations League": 5,
    "CONCACAF Nations League": 536, "Eliminatorias CONMEBOL": 34,
    "Eliminatorias UEFA": 32, "Eliminatorias CONCACAF": 33,
    "Eliminatorias AFC": 35, "Eliminatorias CAF": 29, "COPA ASIA": 1008,
    "COPA AFRICA": 6, "Friendlies": 10
}

TEMPORADAS = ["2023", "2024", "2025", "2026"]
LIMITE_DIARIO = 90
DB_PATH = 'data/futbol.db'
CSV_ANTIGUO = 'datos_reales_selecciones.csv'

def conectar_db():
    # Nos aseguramos de que la carpeta data exista
    os.makedirs('data', exist_ok=True)
    return sqlite3.connect(DB_PATH)

def migrar_csv_a_db(conn):
    # Si existe el CSV antiguo, lo pasamos a la base de datos SQLite
    if os.path.exists(CSV_ANTIGUO):
        print(f"🔄 Migrando datos del CSV antiguo ({CSV_ANTIGUO}) a SQLite...")
        df_csv = pd.read_csv(CSV_ANTIGUO)
        df_csv.to_sql('partidos', conn, if_exists='append', index=False)
        # Renombramos el CSV para no volver a migrarlo
        os.rename(CSV_ANTIGUO, CSV_ANTIGUO + ".backup")
        print("✅ Migración completada. CSV renombrado a .backup")

def extraer_partidos():
    conn = conectar_db()
    peticiones_hoy = 0
    ids_procesados = set()

    # Comprobar si la tabla 'partidos' existe y cargar IDs previos
    try:
        df_previo = pd.read_sql("SELECT fixture_id FROM partidos", conn)
        ids_procesados = set(df_previo['fixture_id'].tolist())
        print(f"🗄️ Base de datos conectada. {len(ids_procesados)} partidos ya procesados anteriormente.")
    except sqlite3.OperationalError:
        print("🗄️ Base de datos nueva. Comprobando si hay datos antiguos para migrar...")
        migrar_csv_a_db(conn)
        try:
            df_previo = pd.read_sql("SELECT fixture_id FROM partidos", conn)
            ids_procesados = set(df_previo['fixture_id'].tolist())
        except sqlite3.OperationalError:
            print("Iniciando desde cero.")

    nuevos_datos = []
    limite_alcanzado = False

    for temp in TEMPORADAS:
        if limite_alcanzado: break
            
        for nombre_comp, liga_id in COMPETICIONES.items():
            if limite_alcanzado: break
                
            print(f"Buscando: {nombre_comp} ({temp})...")
            url_fixtures = "https://v3.football.api-sports.io/fixtures"
            querystring = {"league": str(liga_id), "season": temp}
            
            try:
                res = requests.get(url_fixtures, headers=HEADERS, params=querystring)
                peticiones_hoy += 1
                respuesta_json = res.json()
            except Exception as e:
                print(f"🛑 Error de conexión: {e}")
                limite_alcanzado = True
                break

            if 'errors' in respuesta_json and respuesta_json['errors']:
                print(f"🛑 BLOQUEO DE API: {respuesta_json['errors']}")
                limite_alcanzado = True
                break
            
            partidos = respuesta_json.get('response', [])
            time.sleep(7) 
            
            for p in partidos:
                fixture_id = p['fixture']['id']
                
                # Saltar si el partido no ha terminado o ya lo tenemos en SQLite
                if p['fixture']['status']['short'] not in ['FT', 'AET', 'PEN'] or fixture_id in ids_procesados:
                    continue
                
                if peticiones_hoy >= LIMITE_DIARIO:
                    print("\n⚠️ LÍMITE DIARIO ALCANZADO. Deteniendo por hoy.")
                    limite_alcanzado = True
                    break

                home_team = p['teams']['home']['name']
                away_team = p['teams']['away']['name']
                
                print(f"[{peticiones_hoy}/{LIMITE_DIARIO}] Descargando stats: {home_team} vs {away_team}")
                
                url_stats = "https://v3.football.api-sports.io/fixtures/statistics"
                try:
                    res_stats = requests.get(url_stats, headers=HEADERS, params={"fixture": str(fixture_id)})
                    peticiones_hoy += 1
                    respuesta_stats_json = res_stats.json()
                except Exception as e:
                    print(f"🛑 Error de red en stats: {e}")
                    limite_alcanzado = True
                    break
                
                stats_data = respuesta_stats_json.get('response', [])
                
                if not stats_data or len(stats_data) < 2:
                    time.sleep(7)
                    continue
                
                def get_stat(s_list, tipo):
                    for item in s_list:
                        if item['type'] == tipo and item['value'] is not None:
                            return int(item['value'])
                    return 0

                h_s = get_stat(stats_data[0]['statistics'], "Total Shots")
                h_st = get_stat(stats_data[0]['statistics'], "Shots on Goal")
                h_c = get_stat(stats_data[0]['statistics'], "Corner Kicks")
                h_y = get_stat(stats_data[0]['statistics'], "Yellow Cards") + get_stat(stats_data[0]['statistics'], "Red Cards")
                
                a_s = get_stat(stats_data[1]['statistics'], "Total Shots")
                a_st = get_stat(stats_data[1]['statistics'], "Shots on Goal")
                a_c = get_stat(stats_data[1]['statistics'], "Corner Kicks")
                a_y = get_stat(stats_data[1]['statistics'], "Yellow Cards") + get_stat(stats_data[1]['statistics'], "Red Cards")

                nuevos_datos.append({
                    'fixture_id': fixture_id,
                    'date': p['fixture']['date'][:10],
                    'tournament': nombre_comp,
                    'home_team': home_team, 'away_team': away_team,
                    'home_score': p['goals']['home'], 'away_score': p['goals']['away'],
                    'HS': h_s, 'AS': a_s, 'HST': h_st, 'AST': a_st,
                    'HC': h_c, 'AC': a_c, 'HY': h_y, 'AY': a_y,
                    'neutral': False
                })
                time.sleep(7)

    # Si hay datos nuevos, los guardamos en SQLite
    if nuevos_datos:
        df_final = pd.DataFrame(nuevos_datos)
        df_final.to_sql('partidos', conn, if_exists='append', index=False)
        print(f"\n✅ Se añadieron {len(nuevos_datos)} partidos nuevos a la base de datos.")
    else:
        print("\n✅ No hay partidos nuevos para descargar hoy.")
        
    conn.close()

if __name__ == "__main__":
    extraer_partidos()
