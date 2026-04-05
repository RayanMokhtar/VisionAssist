"""
MQTTBroker – Implémentation MQTT via paho-mqtt.

Compatible Mosquitto local sur Jetson (ou tout broker MQTT 3.1.1 / 5.0).

Installation : pip install paho-mqtt>=2.0.0
Broker local  : sudo apt install mosquitto mosquitto-clients
                sudo systemctl enable --now mosquitto
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

import paho.mqtt.client as mqtt

import configuration as cfg
from broker.broker_interface import IBroker

logger = logging.getLogger(__name__)


class MQTTBroker(IBroker):
    """
    Broker MQTT basé sur paho-mqtt.

    Fonctionnalités :
        • Connexion/reconnexion automatique
        • Publication JSON sérialisée
        • Abonnements multiples avec pattern matching (#, +)
        • Réabonnement automatique après reconnexion
        • Thread-safe

    Exemple d'utilisation :
        broker = MQTTBroker()
        broker.connect()

        broker.publish("visionassist/stt/result", {"text": "Bonjour"})

        def on_message(topic, payload):
            print(f"Reçu sur {topic} : {payload}")

        broker.subscribe("visionassist/#", on_message)
    """

    def __init__(
        self,
        host: str = cfg.BROKER_HOST,
        port: int = cfg.BROKER_PORT,
        client_id: str = cfg.BROKER_CLIENT_ID,
        username: Optional[str] = cfg.BROKER_USERNAME,
        password: Optional[str] = cfg.BROKER_PASSWORD,
        keepalive: int = cfg.BROKER_KEEPALIVE,
    ) -> None:
        self._host = host
        self._port = port
        self._keepalive = keepalive
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}
        self._lock = threading.Lock()

        # Initialisation client paho (protocol v3.1.1)
        self._client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )
        self._client.enable_logger(logger)

        if username:
            self._client.username_pw_set(username, password)

        if cfg.BROKER_USE_TLS:
            self._client.tls_set()

        # Callbacks paho internes
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Connexion asynchrone + démarrage boucle réseau en arrière-plan."""
        self._client.connect(self._host, self._port, self._keepalive)
        self._client.loop_start()   # thread daemon paho (non-bloquant)
        logger.info("MQTT : connexion initialisée → %s:%d", self._host, self._port)

    def disconnect(self) -> None:
        """Arrêt propre de la boucle et déconnexion."""
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("MQTT : déconnecté.")

    # ─── Publish ──────────────────────────────────────────────────────────

    def publish(
        self,
        topic: str,
        payload: Any,
        qos: int = cfg.BROKER_QOS,
        retain: bool = cfg.BROKER_RETAIN,
    ) -> bool:
        """
        Publie un message sur un topic MQTT.

        - dict/list → sérialisé en JSON automatiquement
        - Attend l'accusé de réception (wait_for_publish, 5s max)
        """
        try:
            if isinstance(payload, (dict, list)):
                payload_str = json.dumps(payload, ensure_ascii=False)
            else:
                payload_str = str(payload)

            result = self._client.publish(topic, payload_str, qos=qos, retain=retain)
            result.wait_for_publish(timeout=5.0)

            preview = payload_str[:100] + "…" if len(payload_str) > 100 else payload_str
            logger.debug("MQTT ↑ [%s] %s", topic, preview)
            return result.rc == mqtt.MQTT_ERR_SUCCESS

        except Exception as exc:
            logger.error("MQTT publish échoué sur '%s' : %s", topic, exc)
            return False

    # ─── Subscribe ────────────────────────────────────────────────────────

    def subscribe(
        self,
        topic: str,
        callback: Callable[[str, Any], None],
        qos: int = cfg.BROKER_QOS,
    ) -> None:
        """
        S'abonne à un topic (ou pattern + / #) avec un callback.

        Le callback reçoit (topic: str, payload: dict | str).
        """
        with self._lock:
            self._subscriptions[topic] = callback
        self._client.subscribe(topic, qos)
        logger.info("MQTT : abonné à '%s'", topic)

    def unsubscribe(self, topic: str) -> None:
        """Se désabonne et supprime le callback."""
        with self._lock:
            self._subscriptions.pop(topic, None)
        self._client.unsubscribe(topic)
        logger.info("MQTT : désabonné de '%s'", topic)

    # ─── Status ───────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._connected

    # ─── Callbacks paho (internes) ────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc: int) -> None:
        _CODES = {
            0: "Connexion acceptée ✓",
            1: "Version protocole refusée",
            2: "Identifiant client refusé",
            3: "Serveur indisponible",
            4: "Identifiants incorrects",
            5: "Non autorisé",
        }
        if rc == 0:
            self._connected = True
            logger.info("MQTT connecté : %s", _CODES.get(rc, f"code {rc}"))
            # Réabonnement automatique après reconnexion
            with self._lock:
                for topic in list(self._subscriptions):
                    self._client.subscribe(topic)
        else:
            logger.error("MQTT connexion refusée : %s", _CODES.get(rc, f"code {rc}"))

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        self._connected = False
        if rc != 0:
            logger.warning(
                "MQTT déconnexion inattendue (rc=%d) — reconnexion automatique…", rc
            )
        else:
            logger.info("MQTT : déconnexion propre.")

    def _on_message(self, client, userdata, msg) -> None:
        """Dispatch d'un message MQTT vers le bon callback."""
        topic = msg.topic
        try:
            raw = msg.payload.decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
        except Exception as exc:
            logger.error("MQTT : décodage impossible sur '%s' : %s", topic, exc)
            return

        # Recherche du callback (exact ou pattern)
        with self._lock:
            callback = None
            for sub_topic, cb in self._subscriptions.items():
                if mqtt.topic_matches_sub(sub_topic, topic):
                    callback = cb
                    break

        if callback:
            try:
                callback(topic, payload)
            except Exception as exc:
                logger.error("MQTT : erreur callback sur '%s' : %s", topic, exc)
        else:
            logger.debug("MQTT ↓ [%s] — aucun handler enregistré", topic)
