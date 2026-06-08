from sqlalchemy import create_engine, engine
from sqlalchemy.orm import sessionmaker

from configuration import CONFIGURATION
from persistance.models import Base

def initialiser_session_db():
    def get_url():
        db_conf = CONFIGURATION.db

        if db_conf.url is not None : 
            return db_conf.url

        if db_conf.dialecte == "sqlite":
            return f"sqlite:///{db_conf.db_file}"

        raise ValueError(f" erreur conf db config (seul sqlite est supporté): {db_conf}")
    
    url = get_url()  
    print("Using DB URL:", url)  
    engine = create_engine(
        url,
        echo=CONFIGURATION.db.echo,
    )
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
    return engine, SessionLocal()

def creer_tables():
    engine, session_id = initialiser_session_db()
    Base.metadata.create_all(bind=engine)
    return engine, session_id 


engine_sortie , SESSION_ID  = creer_tables() 