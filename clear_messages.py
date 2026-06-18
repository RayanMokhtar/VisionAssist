import sqlite3

def clear_messages():
    try:
        # Connexion à la base de données SQLite
        conn = sqlite3.connect('db_file.db')
        cursor = conn.cursor()
        
        # Exécution de la requête SQL pour vider la table messages
        cursor.execute("DELETE FROM messages;")
        
        # Validation des changements
        conn.commit()
        
        print(f"Succès : {cursor.rowcount} messages ont été supprimés de la base de données.")
        
    except sqlite3.Error as e:
        print(f"Erreur lors de la suppression : {e}")
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_messages()
