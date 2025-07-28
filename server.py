import http.server

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        return super(CORSRequestHandler, self).end_headers()

if __name__ == '__main__':
    http.server.test(HandlerClass=CORSRequestHandler, port=8013)
