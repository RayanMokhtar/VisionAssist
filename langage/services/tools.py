"""
Tools — Outils invocables par l'agent LLM via tool calling.

Chaque outil est décoré avec @tool (LangChain) et sera automatiquement
exposé au LLM via son JSON Schema. Le LLM décide quand les appeler.

Outils disponibles :
    - get_current_time    : date et heure actuelle
    - get_weather         : météo via OpenMeteo
    - search_nearby_place : recherche de lieu via Nominatim/OSM
    - get_transit_info    : horaires SNCF/IDF
    - query_memory        : recherche RAG dans les messages de conversation
"""

from __future__ import annotations

import datetime as _dt
import logging
import subprocess
from typing import List, Annotated

import requests
from langage.services.model import MODEL_SERVICE

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import os
from langage.database.vector_store import vector_store
from persistance.repository import REPOSITORIES

logger = logging.getLogger(__name__)



def _log_tool(tool_name: str, result: str) -> None:
    """Affiche un log structuré quand un outil est invoqué."""
    preview = result[:200].replace("\n", " ")
    logger.info("🔧 [TOOL] %-25s → %s", tool_name, preview)




@tool
def get_current_time() -> str:
    """Retourne la date et l'heure EXACTE et ACTUELLE en français.
    IMPORTANT : Tu DOIS appeler cet outil à chaque fois que l'utilisateur demande
    l'heure ou la date. Ne jamais utiliser une heure mémorisée d'un échange précédent.
    L'heure change à chaque seconde — toujours appeler cet outil pour avoir la valeur fraîche.
    """
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    mois = [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre"
    ]

    try:
        raw = subprocess.check_output(["date", "+%H:%M:%S|%w|%d|%m|%Y"], text=True).strip()
        heure, minute, seconde = raw.split("|")[0].split(":")
        weekday_num = int(raw.split("|")[1])
        jour = int(raw.split("|")[2])
        mois_num = int(raw.split("|")[3])
        annee = raw.split("|")[4]
        
        jour_nom = jours[(weekday_num - 1) % 7]
        mois_nom = mois[mois_num - 1]
        result = f"Il est {int(heure)}h{minute}. Nous sommes {jour_nom} {jour} {mois_nom} {annee}."
    except Exception as e:
        logger.warning("Fallback datetime.now() car subprocess.date a échoué: %s", e)
        now = _dt.datetime.now()
        jour_nom = jours[now.weekday()]
        mois_nom = mois[now.month - 1]
        result = f"Il est {now.hour}h{now.minute:02d}. Nous sommes {jour_nom} {now.day} {mois_nom} {now.year}."

    logger.info(f"⏰ [TOOL APPELÉ] get_current_time → {result}")
    _log_tool("get_current_time", result)
    return result


@tool
def get_weather(city: str = "Paris") -> str:
    """Obtient la météo actuelle pour une ville donnée.
    Utile quand l'utilisateur demande le temps qu'il fait, s'il va pleuvoir,
    ou la température.

    Args:
        city: Nom de la ville (en français).
    """
    logger.info("🌤️  [TOOL] get_weather appelé avec city='%s'", city)
    try:
        # Géocodage de la ville
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "fr"},
            timeout=5,
        )
        geo_data = geo_resp.json()
        if not geo_data.get("results"):
            result = f"Ville '{city}' non trouvée."
            _log_tool("get_weather", result)
            return result

        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", city)

        # Météo actuelle
        meteo_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "timezone": "Europe/Paris",
            },
            timeout=5,
        )
        meteo = meteo_resp.json().get("current_weather", {})

        temp = meteo.get("temperature", "?")
        wind = meteo.get("windspeed", "?")
        code = meteo.get("weathercode", -1)

        weather_codes = {
            0: "ciel dégagé", 1: "principalement dégagé", 2: "partiellement nuageux",
            3: "couvert", 45: "brouillard", 48: "brouillard givrant",
            51: "bruine légère", 53: "bruine", 55: "bruine forte",
            61: "pluie légère", 63: "pluie modérée", 65: "pluie forte",
            71: "neige légère", 73: "neige modérée", 75: "neige forte",
            80: "averses légères", 81: "averses", 82: "averses fortes",
            95: "orage", 96: "orage avec grêle", 99: "orage violent avec grêle",
        }
        description = weather_codes.get(code, "conditions inconnues")

        result = (
            f"Météo à {city_name} : {description}, "
            f"température {temp}°C, vent {wind} km/h."
        )
        _log_tool("get_weather", result)
        return result

    except requests.RequestException as e:
        logger.error("Erreur API météo : %s", e)
        result = "Impossible d'obtenir la météo pour le moment."
        _log_tool("get_weather", result)
        return result


@tool
def search_nearby_place(query: str, latitude: float, longitude: float) -> str:
    """Recherche un lieu ou commerce à proximité de la position actuelle de l'utilisateur.
    Utile quand l'utilisateur cherche un café, restaurant, pharmacie,
    arrêt de bus, école, ou tout autre lieu proche.

    Args:
        query: Ce que l'utilisateur recherche (ex: "pharmacie", "café", "boulangerie").
        latitude: Latitude GPS de l'utilisateur (fournie dans le message).
        longitude: Longitude GPS de l'utilisateur (fournie dans le message).
    """
    lat = latitude
    lon = longitude

    logger.info("📍 [TOOL] search_nearby_place | query='%s' | position=(%.4f, %.4f)", query, lat, lon)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": 5,
                "addressdetails": 1,
                "accept-language": "fr",
                "viewbox": f"{lon-0.02},{lat+0.02},{lon+0.02},{lat-0.02}",
                "bounded": 1,
            },
            headers={"User-Agent": "VisionAssist/1.0 (assistant-malvoyant)"},
            timeout=5,
        )
        results = resp.json()

        if not results:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query,
                    "format": "json",
                    "limit": 3,
                    "addressdetails": 1,
                    "accept-language": "fr",
                },
                headers={"User-Agent": "VisionAssist/1.0"},
                timeout=5,
            )
            results = resp.json()

        if not results:
            result = f"Aucun résultat trouvé pour '{query}' à proximité."
            _log_tool("search_nearby_place", result)
            return result

        lines = []
        for r in results[:3]:
            name = r.get("display_name", "Lieu inconnu")
            parts = name.split(",")
            short_name = ", ".join(parts[:3]).strip()
            lines.append(f"- {short_name}")

        result = f"Lieux trouvés pour '{query}' :\n" + "\n".join(lines)
        _log_tool("search_nearby_place", result)
        return result

    except requests.RequestException as e:
        logger.error("Erreur recherche lieu : %s", e)
        result = "Impossible d'effectuer la recherche pour le moment."
        _log_tool("search_nearby_place", result)
        return result


@tool
def query_memory(query: str, days_back: int = 7, state: Annotated[dict, InjectedState] = None) -> str:
    """Recherche dans la mémoire conversationnelle de l'utilisateur.
    Utilise la recherche sémantique (RAG) sur les messages des sessions passées.
    Utile quand l'utilisateur demande 'qu'est-ce qu'on a dit hier ?',
    'rappelle-moi qui m'a demandé un stylo', 'on avait parlé de quoi ?'.

    Args:
        query: La question ou le sujet à rechercher dans les conversations passées.
        days_back: Nombre de jours en arrière à parcourir (défaut 7). Si l'utilisateur mentionne 'il y a 3 jours', mettre 3.
    """
    import datetime
    logger.info("[TOOL] query_memory : query='%s' days_back=%d", query, days_back)
    user_id = str((state or {}).get("user_id", "default_user"))
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days_back)

    try:
        # Recherche sémantique FAISS — filtre user_id dans les métadonnées
        raw_results = vector_store.search(
            collection_name="conversation_messages",
            query=query,
            n_results=20,  # On en récupère plus pour filtrer ensuite par date
            where={"user_id": user_id},
        )

        # Filtrer côté Python sur la fenêtre temporelle
        results = []
        for r in raw_results:
            ts_str = r.get("metadata", {}).get("timestamp", "")
            try:
                ts = datetime.datetime.fromisoformat(ts_str) if ts_str else datetime.datetime.min
            except ValueError:
                ts = datetime.datetime.min
            if ts >= cutoff:
                results.append(r)

        results = results[:5]  # Garder les 5 meilleurs (déjà triés par pertinence FAISS)

        if not results:
            result = f"Aucun échange pertinent trouvé pour '{query}' dans les {days_back} derniers jours."
            _log_tool("query_memory", result)
            return result

        # Formater les résultats pour le LLM
        lines = []
        for i, r in enumerate(results, 1):
            date = r.get("metadata", {}).get("timestamp", "")[:16].replace("T", " ")
            score = 1.0 - r.get("distance", 1.0)
            prefix = f"[{date}] (pertinence: {score:.0%})" if date else f"(pertinence: {score:.0%})"
            lines.append(f"{i}. {prefix} {r['content'][:400]}")

        result = "Voici les échanges passés les plus pertinents :\n" + "\n".join(lines)
        _log_tool("query_memory", result)
        return result

    except Exception as e:
        logger.error("Erreur recherche mémoire : %s", e)
        result = "Impossible d'accéder à la mémoire pour le moment."
        _log_tool("query_memory", result)
        return result

@tool
def besoin_image(instruction: str, state: Annotated[dict, InjectedState] = None) -> str:
    """Utilise cet outil UNIQUEMENT si tu dois analyser ce qu'il y a devant l'utilisateur via la caméra.
    
    IMPORTANT : Tu dois me transmettre l'instruction exacte de ce que le module vision doit chercher pour l'utilisateur.
    Le module vision répondra DIRECTEMENT à l'utilisateur, ce sera le dernier mot de la conversation.
    """
    image_url = (state or {}).get("pending_image_url")
    if not image_url:
        result = "Aucune image de la caméra n'est disponible. Demande à l'utilisateur d'activer sa caméra."
        _log_tool("besoin_image", result)
        return result
        
    messages = (state or {}).get("messages", [])
    user_query = "la demande de l'utilisateur"
    # Recherche du dernier message humain dans l'historique
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human":
            user_query = msg.content
            break
            
    from langage.services.model import MODEL_SERVICE
    logger.info("👁️ [VISION] Génération de la réponse finale basée sur : %s", instruction)
    
    # Le prompt pour le VLM lui demande d'agir comme l'assistant final sans réfléchir à voix haute
    prompt = (
        f"L'utilisateur te demande : '{user_query}'.\n"
        f"Consigne d'analyse : '{instruction}'.\n"
        "IMPORTANT: Tu es un assistant vocal. Tu vas sûrement faire un raisonnement interne, mais "
        "tu DOIS OBLIGATOIREMENT placer la phrase finale que le synthétiseur vocal prononcera "
        "entre les balises <REPONSE> et </REPONSE>.\n"
        "Exemple: <REPONSE>Bonjour, je vois une boîte de médicaments avec écrit Aspirine.</REPONSE>\n"
        "Ne mets rien d'autre dans ces balises que le texte à prononcer."
    )
    
    try:
        # Appel direct au modèle de vision pour générer la réponse finale
        description = MODEL_SERVICE.poser_question_sur_image(prompt, image_url)
        result = description
    except Exception as e:
        logger.error("Erreur lors de l'analyse visuelle : %s", e)
        result = "Une erreur est survenue lors de l'analyse de l'image."
        
    _log_tool("besoin_image", result)
    return result

@tool
def get_transit_info(destination: str, latitude: float = None, longitude: float = None, origin: str = "") -> str:
    """Donne les prochains itinéraires de transport en commun (train, RER, métro, bus)
    vers une destination.
    Si l'utilisateur spécifie explicitement un point de départ (ex: "à partir de Paris Saint Lazare"), 
    tu DOIS le mettre dans 'origin'. S'il ne précise rien, laisse 'origin' vide et sa position actuelle sera utilisée.
    Utile quand l'utilisateur demande 'quand est le prochain RER ?',
    'comment aller à Paris ?', 'quel train pour [destination] depuis [origine] ?'.

    Args:
        destination: La destination souhaitée (ex: "Paris Gare du Nord", "Châtelet", "Versailles").
        latitude: Latitude GPS de l'utilisateur (fournie dans le message). Optionnelle si origin est rempli.
        longitude: Longitude GPS de l'utilisateur (fournie dans le message). Optionnelle si origin est rempli.
        origin: Le point de départ explicite, uniquement si demandé. Sinon vide.
    """

    logger.info("🚆 [TOOL] get_transit_info | destination='%s' origin='%s'", destination, origin)

    # Récupérer la clé API
    api_key = os.environ.get("SNCF_API_KEY", "53460872-dc6a-41f4-8197-0d00ba8e2074")
    if not api_key:
        result = "La clé API SNCF n'est pas configurée. Impossible d'obtenir les horaires de transport."
        _log_tool("get_transit_info", result)
        return result

    lat = latitude
    lon = longitude

    # On essaie d'abord fr-idf (réseau IDF), puis la coverage nationale SNCF
    COVERAGES = [
        "https://api.sncf.com/v1/coverage/fr-idf",
        "https://api.sncf.com/v1/coverage/sncf",
    ]
    auth = (api_key, "")  # Basic Auth : clé = username, mdp vide
    headers = {"Accept": "application/json"}
    BASE_URL = COVERAGES[0]

    def search_navitia_place(base_url: str, search_query: str):
        """Recherche un lieu sur Navitia, retourne la liste de places."""
        resp = requests.get(
            f"{base_url}/places",
            params={"q": search_query, "count": 10},
            auth=auth,
            headers=headers,
            timeout=6,
        )
        raw = resp.json()
        # Log complet pour diagnostic
        logger.info(
            "🚆 [NAVITIA] /places?q='%s' → HTTP %s, places=%d, error=%s",
            search_query, resp.status_code,
            len(raw.get("places", [])),
            raw.get("error", {}).get("message", "none")
        )
        return raw.get("places", [])

    def extract_dest(places: list):
        """Extrait le meilleur résultat (stop_area > stop_point > autre)."""
        for p in places:
            if p.get("embedded_type") in ["stop_area", "stop_point"]:
                # Navitia encapsule les données dans p[embedded_type]
                embedded = p.get(p["embedded_type"], {})
                return {
                    "id": embedded.get("id") or p.get("id"),
                    "name": embedded.get("name") or p.get("name", "?"),
                }
        if places:
            p = places[0]
            embedded = p.get(p.get("embedded_type", ""), {})
            return {
                "id": embedded.get("id") or p.get("id"),
                "name": embedded.get("name") or p.get("name", "?"),
            }
        return None

    try:
        # ── Étape 1 : Résoudre la destination en ID Navitia ──────────────────
        dest_place = None
        used_base_url = COVERAGES[0]

        for coverage_url in COVERAGES:
            places = search_navitia_place(coverage_url, destination)
            # Variantes si aucun résultat
            if not places:
                places = search_navitia_place(coverage_url, destination.replace(" ", "-"))
            if not places and not destination.lower().startswith("gare"):
                places = search_navitia_place(coverage_url, "Gare " + destination)
            dest_place = extract_dest(places)
            if dest_place:
                used_base_url = coverage_url
                break

        if not dest_place:
            result = f"Je n'ai pas trouvé l'arrêt ou la gare '{destination}' dans le réseau SNCF/IDF."
            _log_tool("get_transit_info", result)
            return result

        BASE_URL = used_base_url
        dest_id = dest_place["id"]
        dest_name = dest_place["name"]
        logger.info("🚆 [TRANSIT] Destination résolue : '%s' → %s (%s)", destination, dest_name, dest_id)

        # ── Étape 2 : Résoudre l'origine (position de l'utilisateur ou texte) ──────────
        now_str = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")

        origin_id = None
        if origin and origin.strip():
            # Recherche textuelle de l'origine
            origin_places = search_navitia_place(BASE_URL, origin)
            if not origin_places:
                origin_places = search_navitia_place(BASE_URL, origin.replace(" ", "-"))
            if not origin_places and not origin.lower().startswith("gare"):
                origin_places = search_navitia_place(BASE_URL, "Gare " + origin)
            
            origin_place_data = extract_dest(origin_places)
            if origin_place_data:
                origin_id = origin_place_data["id"]
                logger.info("🚆 [TRANSIT] Origine explicite résolue : '%s' → %s", origin, origin_id)
            else:
                result = f"Je n'ai pas trouvé la gare de départ '{origin}' dans le réseau."
                _log_tool("get_transit_info", result)
                return result
        elif lat is not None and lon is not None:
            # Tentative 1 : places_nearby (rayon 5km pour trouver une gare)
            for radius in [2000, 5000, 10000]:
                try:
                    nearby_resp = requests.get(
                        f"{BASE_URL}/coords/{lon};{lat}/places_nearby",
                        params={"type[]": "stop_area", "count": 3, "radius": radius},
                        auth=auth,
                        headers=headers,
                        timeout=6,
                    )
                    nearby_data = nearby_resp.json()
                    logger.info(
                        "🚆 [ORIGIN] places_nearby radius=%d → HTTP %s, found=%d",
                        radius, nearby_resp.status_code,
                        len(nearby_data.get("places_nearby", []))
                    )
                    nearby_places = nearby_data.get("places_nearby", [])
                    if nearby_places:
                        p = nearby_places[0]
                        embedded = p.get(p.get("embedded_type", ""), {})
                        origin_id = embedded.get("id") or p.get("id")
                        logger.info("🚆 [TRANSIT] Origine résolue (nearby) : %s", origin_id)
                        break
                except Exception as _e:
                    logger.warning("🚆 [TRANSIT] places_nearby r=%d échoué : %s", radius, _e)

            # Tentative 2 : reverse geocode → trouver la ville et chercher une gare textuellement
            if not origin_id:
                try:
                    geo_resp = requests.get(
                        "https://nominatim.openstreetmap.org/reverse",
                        params={"lat": lat, "lon": lon, "format": "json"},
                        headers={"User-Agent": "VisionAssist/1.0"},
                        timeout=5,
                    )
                    city = geo_resp.json().get("address", {}).get("city") or \
                           geo_resp.json().get("address", {}).get("town") or \
                           geo_resp.json().get("address", {}).get("village", "")
                    if city:
                        city_places = search_navitia_place(BASE_URL, city)
                        origin_place = extract_dest(city_places)
                        if origin_place:
                            origin_id = origin_place["id"]
                            logger.info("🚆 [TRANSIT] Origine résolue (ville '%s') : %s", city, origin_id)
                except Exception as _e:
                    logger.warning("🚆 [TRANSIT] Reverse geocode échoué : %s", _e)

        if not origin_id:
            result = (
                "Je n'arrive pas à localiser une gare SNCF à proximité de votre position. "
                "Essayez de préciser votre gare de départ (ex : 'depuis Cergy-le-Haut')."
            )
            _log_tool("get_transit_info", result)
            return result

        from_param = origin_id

        journeys_resp = requests.get(
            f"{BASE_URL}/journeys",
            params={
                "from": from_param,
                "to": dest_id,
                "datetime": now_str,
                "count": 3,
                "min_nb_journeys": 1,
            },
            auth=auth,
            headers=headers,
            timeout=10,
        )
        journeys_data = journeys_resp.json()
        journeys = journeys_data.get("journeys", [])

        if not journeys:
            error_info = journeys_data.get("error", {})
            logger.warning(
                "🚆 [TRANSIT] Aucun trajet. HTTP=%s from='%s' to='%s' error=%s",
                journeys_resp.status_code, from_param, dest_id,
                error_info.get("message", "none")
            )
            result = f"Aucun trajet trouvé vers '{dest_name}' pour le moment."
            _log_tool("get_transit_info", result)
            return result

        # ── Étape 3 : Formater les résultats pour la voix ────────────────────
        now = _dt.datetime.now()
        lines_out = [f"Prochains trajets vers {dest_name} :"]

        for i, journey in enumerate(journeys[:3], 1):
            sections = journey.get("sections", [])

            # Calculer départ et arrivée globaux
            departure_str = journey.get("departure_date_time", "")
            arrival_str = journey.get("arrival_date_time", "")

            if departure_str:
                dep_dt = _dt.datetime.strptime(departure_str, "%Y%m%dT%H%M%S")
                arr_dt = _dt.datetime.strptime(arrival_str, "%Y%m%dT%H%M%S")
                wait_min = int((dep_dt - now).total_seconds() // 60)
                duration_min = int((arr_dt - dep_dt).total_seconds() // 60)
                dep_hm = dep_dt.strftime("%Hh%M")
                arr_hm = arr_dt.strftime("%Hh%M")
            else:
                dep_hm = arr_hm = "?"
                wait_min = duration_min = 0

            # Extraire les lignes de transport utilisées (pas les marche à pied)
            transport_sections = [
                s for s in sections
                if s.get("type") == "public_transport"
            ]

            if not transport_sections:
                continue

            # Résumer les lignes empruntées
            lines_used = []
            for sec in transport_sections:
                display = sec.get("display_informations", {})
                line_name = display.get("label", display.get("name", ""))
                commercial_mode = display.get("commercial_mode", "")
                direction = display.get("direction", "")
                if line_name:
                    desc = f"{commercial_mode} {line_name}".strip()
                    if direction:
                        desc += f" direction {direction}"
                    lines_used.append(desc)

            lines_summary = " puis ".join(lines_used) if lines_used else "transport en commun"

            if wait_min <= 0:
                timing = "départ immédiat"
            elif wait_min == 1:
                timing = "dans 1 minute"
            else:
                timing = f"dans {wait_min} minutes"

            lines_out.append(
                f"Trajet {i} : {timing} ({dep_hm}), "
                f"arrivée {arr_hm} — {duration_min} min — {lines_summary}."
            )

        if len(lines_out) == 1:
            result = f"Aucun trajet en transport en commun trouvé vers '{dest_name}'."
        else:
            result = "\n".join(lines_out)

        _log_tool("get_transit_info", result)
        return result

    except requests.RequestException as e:
        logger.error("Erreur API SNCF : %s", e)
        result = "Impossible de contacter le service des horaires de transport pour le moment."
        _log_tool("get_transit_info", result)
        return result
    except Exception as e:
        logger.exception("Erreur inattendue dans get_transit_info")
        result = "Une erreur est survenue lors de la récupération des horaires."
        _log_tool("get_transit_info", result)
        return result



def get_all_tools() -> List:
    """Retourne la liste de tous les outils disponibles pour l'agent."""
    return [
        get_current_time,
        get_weather,
        search_nearby_place,
        get_transit_info,
        query_memory,
        besoin_image,
    ]
