"""
Tools — Outils invocables par l'agent LLM via tool calling.

Chaque outil est décoré avec @tool (LangChain) et sera automatiquement
exposé au LLM via son JSON Schema. Le LLM décide quand les appeler.

Contexte utilisateur : les tools accèdent au user_id et session_id
courants via les contextvars définis ci-dessous.

Outils disponibles :
    - get_current_time       : date et heure actuelle (Paris, via API)
    - get_weather            : météo via OpenMeteo (gratuit, pas de clé API)
    - search_nearby_place    : recherche de lieu via Nominatim/OSM (gratuit)
    - get_transit_info       : horaires de transport en commun SNCF (RER, Train, Métro)
    - save_note              : sauvegarde une note/rappel en BDD
    - query_memory           : recherche RAG dans la mémoire long terme
    - describe_current_scene : description visuelle (dépend du module vision)
    - read_text_in_scene     : OCR sur la scène (dépend du module vision)
"""

from __future__ import annotations

import contextvars
import datetime
import logging
from typing import List

import requests
from langchain_core.tools import tool

import os
from langage.database.engine import get_db
from langage.database.repositories import note_repo
from langage.database.vector_store import vector_store
from langage.memory.long_term import long_term_memory

logger = logging.getLogger(__name__)

# ─── Contexte utilisateur (thread-safe via contextvars) ───────────────────────
# Défini par l'agent avant chaque invocation de l'agent LangGraph.

current_user_id = contextvars.ContextVar("current_user_id", default="default")
current_session_id = contextvars.ContextVar("current_session_id", default="")

# Note : les coordonnées GPS ne sont plus dans des contextvars.
# Elles sont injectées dans le texte du message utilisateur et passées
# directement par le LLM comme arguments aux outils.


# ─── Helper log ───────────────────────────────────────────────────────────────

def _log_tool(tool_name: str, result: str) -> None:
    """Affiche un log structuré quand un outil est invoqué."""
    preview = result[:200].replace("\n", " ")
    logger.info("🔧 [TOOL] %-25s → %s", tool_name, preview)


# ─── Outils ───────────────────────────────────────────────────────────────────

@tool
def get_current_time() -> str:
    """Retourne la date et l'heure EXACTE et ACTUELLE en français.
    IMPORTANT : Tu DOIS appeler cet outil à chaque fois que l'utilisateur demande
    l'heure ou la date. Ne jamais utiliser une heure mémorisée d'un échange précédent.
    L'heure change à chaque seconde — toujours appeler cet outil pour avoir la valeur fraîche.
    """
    import subprocess
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    mois = [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre"
    ]

    # Lire l'heure système via la commande 'date' (même source que le terminal WSL)
    try:
        raw = subprocess.check_output(["date", "+%H:%M:%S|%w|%d|%m|%Y"], text=True).strip()
        parts = raw.split("|")
        heure, minute, seconde = parts[0].split(":")
        weekday_num = int(parts[1])  # 0=dimanche, 1=lundi...
        jour = int(parts[2])
        mois_num = int(parts[3])
        annee = parts[4]
        # Convertir dimanche=0 en index Python (lundi=0)
        weekday_idx = (weekday_num - 1) % 7
        jour_nom = jours[weekday_idx]
        mois_nom = mois[mois_num - 1]
        result = (
            f"Il est {int(heure)}h{minute}. "
            f"Nous sommes {jour_nom} {jour} {mois_nom} {annee}."
        )
    except Exception as e:
        logger.warning("Fallback datetime.now() car subprocess.date a échoué: %s", e)
        now = datetime.datetime.now()
        jour_nom = jours[now.weekday()]
        mois_nom = mois[now.month - 1]
        result = (
            f"Il est {now.hour}h{now.minute:02d}. "
            f"Nous sommes {jour_nom} {now.day} {mois_nom} {now.year}."
        )

    print(f"⏰ [TOOL APPELÉ] get_current_time → {result}", flush=True)
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
def save_note(content: str, category: str = "general") -> str:
    """Sauvegarde une note ou un rappel important pour l'utilisateur.
    Utile quand l'utilisateur dit 'rappelle-moi que...', 'note que...',
    ou 'souviens-toi de...'.

    Args:
        content: Le contenu de la note à sauvegarder.
        category: Catégorie de la note ('rappel', 'lieu', 'contact', 'general').
    """
    logger.info("📝 [TOOL] save_note appelé : category='%s' content='%s'", category, content[:80])
    user_id = current_user_id.get()
    session_id = current_session_id.get()

    try:
        with get_db() as db:
            note = note_repo.add_note(
                db=db,
                user_id=user_id,
                content=content,
                category=category,
                session_id=session_id,
            )
            long_term_memory.index_user_note(
                doc_id=note.id,
                content=content,
                user_id=user_id,
                category=category,
            )

        result = f"Note sauvegardée avec succès : « {content[:80]} »"
        _log_tool("save_note", result)
        return result

    except Exception as e:
        logger.error("Erreur sauvegarde note : %s", e)
        result = "Désolé, impossible de sauvegarder la note pour le moment."
        _log_tool("save_note", result)
        return result


@tool
def query_memory(query: str) -> str:
    """Recherche dans la mémoire long terme de l'utilisateur.
    Utile quand l'utilisateur demande 'qu'est-ce qu'on a fait hier ?',
    'on avait parlé de quoi ?', 'rappelle-moi ce que j'avais dit sur...'.

    Args:
        query: La question ou le sujet à rechercher dans la mémoire.
    """
    logger.info("🧠 [TOOL] query_memory appelé : query='%s'", query)
    user_id = current_user_id.get()

    try:
        results = long_term_memory.search(
            query=query,
            user_id=user_id,
            n_results=5,
        )
        result = long_term_memory.format_search_results(results)
        _log_tool("query_memory", result)
        return result

    except Exception as e:
        logger.error("Erreur recherche mémoire : %s", e)
        result = "Impossible d'accéder à la mémoire pour le moment."
        _log_tool("query_memory", result)
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
    import datetime as _dt

    logger.info("🚆 [TOOL] get_transit_info | destination='%s' origin='%s'", destination, origin)

    # Récupérer la clé API
    api_key = os.environ.get("SNCF_API_KEY", "")
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


# ─── Registre des outils ──────────────────────────────────────────────────────

def get_all_tools() -> List:
    """Retourne la liste de tous les outils disponibles pour l'agent."""
    return [
        get_current_time,
        get_weather,
        search_nearby_place,
        get_transit_info,
        save_note,
        query_memory,
    ]
