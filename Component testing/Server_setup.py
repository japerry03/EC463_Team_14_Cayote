import socket
import sys
import os
from generate_key import generate_private_public_key
from generate_key import serealize
from generate_key import x, key_size



print("Starting Socket opening!")

port = 4444 # Any port value over 1023 are guarenteed not privilaged.
host = "127.0.0.1"


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Open socket s.
    s.bind((host, port)) # Bind a port to be connectable to the host address (localhost for now)
    s.listen() # Listen for any connections.
    connect, address = s.accept()
    with connect:
        print(f"Connected by {address}")
        while True:
            data = connect.recv(1024)
            if not data:
                break
            connect.sendall(data)
            
public_key, private_key = generate_private_public_key(address, host, x, key_size)

