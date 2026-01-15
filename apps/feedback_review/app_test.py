"""
Minimal test - just start a basic HTTP server
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>Databricks App Test - Working!</h1>')

print("Starting server on port 8080...")
server = HTTPServer(('0.0.0.0', 8080), Handler)
server.serve_forever()
