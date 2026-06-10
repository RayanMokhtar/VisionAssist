
from typing import Optional

class SessionAuthentifiee:
    def __init__(
        self, 
        user_id: Optional[str] = None, 
        card_id: Optional[str] = None, 
        access_token: Optional[str] = None, 
        refresh_token: Optional[str] = None, 
        prenom: Optional[str] = None, 
        session_id: Optional[str] = None,
        expire_at: Optional[float] = None
    ):
        self.user_id = user_id
        self.carte_id = card_id  
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.prenom = prenom
        self.session_id = session_id
        self.expire_at = expire_at

