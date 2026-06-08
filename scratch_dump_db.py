import sqlite3

conn = sqlite3.connect("visionassist.db")
cursor = conn.cursor()
cursor.execute("SELECT role, content FROM messages")
messages = cursor.fetchall()

print("Messages in DB:")
for role, content in messages:
    print(f"[{role.upper()}] {repr(content)}")

conn.close()
