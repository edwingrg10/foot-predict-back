"""
SofascoreClient — sesión persistente de Playwright.
Un solo browser compartido por todos los scrapers.
"""
from __future__ import annotations

import time
import threading
import logging
from typing import Optional, Any

log = logging.getLogger("scraper.base")

BASE = "https://www.sofascore.com/api/v1"
UA   = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MIN_DELAY = 0.8   # segundos entre requests (respetar rate limit)


class SofascoreClient:
    """Wrapper sobre Playwright que expone fetch(path) → dict."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._pw      = None
        self._browser = None
        self._ctx     = None
        self._page    = None
        self._last_req = 0.0

    # ── Ciclo de vida ────────────────────────────────────────────────────────
    def start(self):
        """Lanza el browser. Llamar antes de hacer requests."""
        from playwright.sync_api import sync_playwright
        self._pw      = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx     = self._browser.new_context(user_agent=UA, locale="es-ES")
        self._page    = self._ctx.new_page()
        # Visitar la home para obtener cookies de sesión
        try:
            self._page.goto("https://www.sofascore.com/", wait_until="domcontentloaded", timeout=20000)
        except Exception:
            pass
        time.sleep(1)
        log.info("[Client] Browser iniciado y sesión establecida")

    def stop(self):
        """Cierra el browser limpiamente."""
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._page = self._ctx = self._browser = self._pw = None
        log.info("[Client] Browser cerrado")

    def _ensure_alive(self):
        if self._page is None:
            self.start()

    # ── Request principal ────────────────────────────────────────────────────
    def fetch(self, path: str, retries: int = 3) -> Optional[dict]:
        """
        GET {BASE}/{path} usando la sesión del browser.
        Retorna el JSON parseado o None si falla.
        """
        with self._lock:
            self._ensure_alive()

            # Rate limiting
            elapsed = time.time() - self._last_req
            if elapsed < MIN_DELAY:
                time.sleep(MIN_DELAY - elapsed)

            url = f"{BASE}{path}"
            for attempt in range(retries):
                try:
                    result: dict = self._page.evaluate(f"""async () => {{
                        const r = await fetch('{url}', {{
                            headers: {{
                                'Accept': 'application/json',
                                'Origin': 'https://www.sofascore.com',
                                'Referer': 'https://www.sofascore.com/',
                            }}
                        }});
                        if (!r.ok) return {{ __status: r.status }};
                        return await r.json();
                    }}""")
                    self._last_req = time.time()

                    if result is None:
                        log.warning(f"[Client] NULL response: {path}")
                        return None
                    if "__status" in result:
                        status = result["__status"]
                        log.warning(f"[Client] HTTP {status}: {path}")
                        if status == 429:          # rate limit — esperar más
                            time.sleep(5 * (attempt + 1))
                            continue
                        return None

                    return result

                except Exception as e:
                    log.error(f"[Client] Error attempt {attempt+1}: {e}")
                    if attempt < retries - 1:
                        time.sleep(2)
                        # Intentar recargar la página si el browser se colgó
                        try:
                            self._page.reload(timeout=10000)
                        except Exception:
                            self.stop()
                            self.start()

            return None

    # ── Context manager ──────────────────────────────────────────────────────
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# Instancia global compartida
client = SofascoreClient()
