try:
    with open('ui/app.py', 'rb') as f:
        content = f.read()
    
    # Replace \r\n with \n, and any stray \r with \n
    content = content.replace(b'\r\n', b'\n')
    content = content.replace(b'\r', b'\n')
    
    with open('ui/app.py', 'wb') as f:
        f.write(content)
        
    print("Fixed CRLF issues in ui/app.py")
except Exception as e:
    print(f"Error: {e}")
