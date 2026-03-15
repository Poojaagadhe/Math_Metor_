import re

log_path = "data/logs/math_mentor.log"

try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    print("--- LAST 150 LINES OF LOG ---")
    
    for line in lines[-150:]:
        print(line, end="")

except Exception as e:
    print(f"Failed to read log: {e}")
