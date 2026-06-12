import requests
import pandas as pd
import time
import os

# ==============================================================================
# 🤖 BOT EXTRACTOR INTELIGENTE - API-FOOTBALL (DESCARGA EN CUOTAS)
# ==============================================================================

API_KEY = "TU_CLAVE_DE_RAPIDAPI_AQUI" # Reemplaza con tu clave
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

COMPETICIONES = {
    "World Cup": 1,
    "Euro": 4,
    "Copa America": 9,
    "Eliminatorias CONMEBOL": 340,
    "Eliminatorias UEFA": 34
}
TEMPORADAS = ["2023", "2024"]
LIMITE_DIARIO = 90
ARCHIVO_DATOS = 'datos_reales_selecciones.csv'

def extraer_partidos():
    peticiones_hoy = 0
    datos_existentes = []
    ids_procesados = set()

    # Si el archivo ya existe, cargamos lo que ya descargamos para no repetirlo
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
            url_fixtures = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            querystring = {"league": str(liga_id), "season": temp}
            
            # Petición para listar los partidos
            res = requests.get(url_fixtures, headers=HEADERS, params=querystring)
            peticiones_hoy += 1
            partidos = res.json().get('response', [])
            time.sleep(1.5)
            
            for p in partidos:
                fixture_id = p['fixture']['id']
                
                # Saltar si el partido no ha terminado o si ya lo descargamos antes
                if p['fixture']['status']['short'] not in ['FT', 'AET', 'PEN'] or fixture_id in ids_procesados:
                    continue
                
                if peticiones_hoy >= LIMITE_DIARIO:
                    print("\n⚠️ LÍMITE DIARIO DE 90 PETICIONES ALCANZADO. Deteniendo por hoy.")
                    limite_alcanzado = True
                    break

                home_team = p['teams']['home']['name']
                away_team = p['teams']['away']['name']
                
                print(f"[{peticiones_hoy}/{LIMITE_DIARIO}] Descargando stats: {home_team} vs {away_team}")
                
                # Petición para las estadísticas
                url_stats = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
                res_stats = requests.get(url_stats, headers=HEADERS, params={"fixture": str(fixture_id)})
                peticiones_hoy += 1
                stats_data = res_stats.json().get('response', [])
                
                # Variables por defecto
                h_s = h_st = h_c = h_y = a_s = a_st = a_c = a_y = 0
                
                if len(stats_data) == 2:
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
                    'date': p['fixture']['date'],
                    'tournament': nombre_comp,
                    'home_team': home_team, 'away_team': away_team,
                    'home_score': p['goals']['home'], 'away_score': p['goals']['away'],
                    'HS': h_s, 'AS': a_s, 'HST': h_st, 'AST': a_st,
                    'HC': h_c, 'AC': a_c, 'HY': h_y, 'AY': a_y,
                    'neutral': False
                })
                time.sleep(1.5) # Pausa obligatoria para la API

    # Guardar todo junto (Lo viejo + Lo nuevo)
    if nuevos_datos:
        df_final = pd.DataFrame(datos_existentes + nuevos_datos)
        df_final.to_csv(ARCHIVO_DATOS, index=False)
        print(f"\n✅ Se añadieron {len(nuevos_datos)} partidos nuevos. Archivo actualizado.")
    else:
        print("\n✅ No hay partidos nuevos para descargar.")

if __name__ == "__main__":
    extraer_partidos()