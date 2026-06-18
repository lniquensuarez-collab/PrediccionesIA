import requests
import pandas as pd
import time
import os
from datetime import datetime

# ==============================================================================
# 🤖 BOT EXTRACTOR V3.1 - CONEXIÓN DIRECTA (CONTROL DE VELOCIDAD DE 10/MIN)
# ==============================================================================

API_KEY = os.environ.get("RAPIDAPI_KEY") 
HEADERS = {
    "x-apisports-key": API_KEY
}

# IDs Corregidos a la base de datos oficial
COMPETICIONES = {
    "World Cup": 1,
    "Euro": 4,
    "Copa America": 9,
    "Africa Cup of Nations": 6,   # Copa Africana de Naciones (Torneo principal)
    "Eliminatorias CONMEBOL": 34, 
    "Eliminatorias UEFA": 32,
    "Eliminatorias CONCACAF": 33, # Norte y Centroamérica (México, USA, Haití)
    "Eliminatorias AFC": 35,      # Asia (Japón, Corea, Irak)
    "Eliminatorias AFC": 32,       # <- Añadir para Japón, Corea del Sur, Arabia
    "Copa Oro": 22,                # <-- ¡Asegúrate de que esta coma esté aquí!
    "Copa Asiática": 18,            # La última línea no necesita coma
    "Eliminatorias CAF": 29,      # África (Eliminatorias Mundialistas)
    "UEFA Nations League": 5,     # Torneo europeo (Noruega, Suecia, etc.)
    "Friendlies": 10,              # Amistosos Internacionales de la FIFA
}

TEMPORADAS = ["2023", "2024", "2025", "2026"]
LIMITE_DIARIO = 90
ARCHIVO_DATOS = 'datos_reales_selecciones.csv'

def extraer_partidos():
    peticiones_hoy = 0
    datos_existentes = []
    ids_procesados = set()

    if os.path.exists(ARCHIVO_DATOS):
        df_previo = pd.read_csv(ARCHIVO_DATOS)
        datos_existentes = df_previo.to_dict('records')
        if 'fixture_id' in df_previo.columns:
            ids_procesados = set(df_previo['fixture_id'].tolist())
        print(f"📄 Archivo encontrado. {len(ids_procesados)} partidos ya procesados anteriormente.")

    nuevos_datos = []
    limite_alcanzado = False

    for temp in TEMPORADAS:
        if limite_alcanzado: break
            
        for nombre_comp, liga_id in COMPETICIONES.items():
            if limite_alcanzado: break
                
            print(f"Buscando: {nombre_comp} ({temp})...")
            url_fixtures = "https://v3.football.api-sports.io/fixtures"
            querystring = {"league": str(liga_id), "season": temp}
            
            res = requests.get(url_fixtures, headers=HEADERS, params=querystring)
            peticiones_hoy += 1
            
            respuesta_json = res.json()
            if 'errors' in respuesta_json and respuesta_json['errors']:
                if isinstance(respuesta_json['errors'], dict) and 'rateLimit' in respuesta_json['errors']:
                    print(f"🛑 Freno de velocidad activado. Esperando...")
                else:
                    print(f"🛑 BLOQUEO DE API: {respuesta_json['errors']}")
                    limite_alcanzado = True
                    break
            
            partidos = respuesta_json.get('response', [])
            time.sleep(7) # ⬅️ PAUSA DE 7 SEGUNDOS PARA RESPETAR EL LÍMITE
            
            for p in partidos:
                fixture_id = p['fixture']['id']
                
                # Saltar si el partido no ha terminado o ya lo tenemos
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
                res_stats = requests.get(url_stats, headers=HEADERS, params={"fixture": str(fixture_id)})
                peticiones_hoy += 1
                
                respuesta_stats_json = res_stats.json()
                
                if 'errors' in respuesta_stats_json and respuesta_stats_json['errors']:
                    print(f"🛑 Error en API al pedir estadísticas: {respuesta_stats_json['errors']}")
                    limite_alcanzado = True
                    break
                    
                stats_data = respuesta_stats_json.get('response', [])
                
                # Si la API no tiene las estadísticas de este partido, lo saltamos para no guardar "0"
                if not stats_data or len(stats_data) < 2:
                    print(f"   ⚠️ Sin estadísticas detalladas para este partido. Saltando...")
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
                    'date': p['fixture']['date'][:10], # Solo guardamos la fecha YYYY-MM-DD
                    'tournament': nombre_comp,
                    'home_team': home_team, 'away_team': away_team,
                    'home_score': p['goals']['home'], 'away_score': p['goals']['away'],
                    'HS': h_s, 'AS': a_s, 'HST': h_st, 'AST': a_st,
                    'HC': h_c, 'AC': a_c, 'HY': h_y, 'AY': a_y,
                    'neutral': False
                })
                time.sleep(7) # ⬅️ PAUSA DE 7 SEGUNDOS ENTRE CADA PARTIDO

    if nuevos_datos:
        df_final = pd.DataFrame(datos_existentes + nuevos_datos)
        df_final.to_csv(ARCHIVO_DATOS, index=False)
        print(f"\n✅ Se añadieron {len(nuevos_datos)} partidos nuevos. Archivo actualizado.")
    else:
        print("\n✅ No hay partidos nuevos para descargar hoy.")

if __name__ == "__main__":
    extraer_partidos()
