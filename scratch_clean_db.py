import sqlite3

conn = sqlite3.connect("visionassist.db")
cursor = conn.cursor()

# Find messages containing <parameter=
cursor.execute("SELECT id, content FROM messages WHERE content LIKE '%<parameter=%'")
rows = cursor.fetchall()

print(f"Found {len(rows)} malformed messages.")
for row in rows:
    print(f"Deleting message {row[0]}: {row[1]}")
    cursor.execute("DELETE FROM messages WHERE id = ?", (row[0],))

conn.commit()
conn.close()
print("Done.")
