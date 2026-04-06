from abc import ABC, abstractmethod
from typing import Any, Callable


class IBroker(ABC):

    @abstractmethod
    def connexion(self) -> None:
        ...

    @abstractmethod
    def deconnexion(self) -> None:
        """Ferme proprement la connexion."""

    @abstractmethod
    def publier(self,topic: str,payload: Any) -> bool:
        pass


    @abstractmethod
    def sabonner(self , topic: str, fonction_apres_trigger: Callable) -> None:
        ...

    @abstractmethod
    def desabonner(self, topic: str) -> None:
        pass

    @abstractmethod
    def is_connected_to_broker(self) -> bool:
        pass