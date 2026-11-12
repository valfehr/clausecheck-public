"""Minimal HTTP load balancer for OpenAI-compatible model servers (streaming-safe). Costs $0.

  uv run python scripts/lb.py --port 8080 --backends http://127.0.0.1:8081 http://127.0.0.1:8082 --policy least

Policies
- round  : round-robin. Simple, but blind to load: a long generation and a short one count the same.
- least  : least outstanding requests. Sends each request to the replica with the fewest in flight,
           which is what you want when request lengths vary (LLM generations do, a lot).

A teaching tool: in production this is an ALB/NGINX/Envoy (or a gateway such as LiteLLM, or KServe on
Kubernetes) with health checks, retries and autoscaling on queue depth or GPU utilisation.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

LOCK = threading.Lock()
INFLIGHT: dict[str, int] = {}
SERVED: dict[str, int] = {}
BACKENDS: list[str] = []
POLICY = "least"
_rr = None


def pick() -> str:
    with LOCK:
        if POLICY == "round":
            b = next(_rr)
        else:
            b = min(BACKENDS, key=lambda x: INFLIGHT[x])
        INFLIGHT[b] += 1
        SERVED[b] += 1
        return b


def release(b: str) -> None:
    with LOCK:
        INFLIGHT[b] -= 1


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # quiet
        pass

    def _forward(self, method: str) -> None:
        if self.path == "/lb/stats":
            body = repr({"policy": POLICY, "inflight": INFLIGHT, "served": SERVED}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        backend = pick()
        try:
            u = urlparse(backend)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Backend", backend)
            self.end_headers()
            while chunk := resp.read1(8192):  # stream through as it arrives (SSE tokens)
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            conn.close()
        finally:
            release(backend)

    def do_POST(self) -> None:
        self._forward("POST")

    def do_GET(self) -> None:
        self._forward("GET")


def main() -> None:
    global POLICY, _rr
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--backends", nargs="+", required=True)
    ap.add_argument("--policy", choices=["round", "least"], default="least")
    a = ap.parse_args()
    POLICY = a.policy
    BACKENDS.extend(a.backends)
    INFLIGHT.update({b: 0 for b in BACKENDS})
    SERVED.update({b: 0 for b in BACKENDS})
    _rr = itertools.cycle(BACKENDS)
    print(f"LB on :{a.port} → {BACKENDS} ({POLICY})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Proxy).serve_forever()


if __name__ == "__main__":
    main()
