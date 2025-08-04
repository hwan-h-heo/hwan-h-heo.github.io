import http.server

class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        # COEP 정책을 credentialless로 완화
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

if __name__ == '__main__':
    http.server.test(HandlerClass=MyRequestHandler, port=8013)