import sqlite3
conn = sqlite3.connect('data/memory.db')
cursor = conn.cursor()
cursor.execute("SELECT extracted_text, solution FROM problems WHERE solution LIKE '%u_%' OR extracted_text LIKE '%u_%'")
for row in cursor.fetchall():
    print(row)
