"""
OceanEmbed (SIH 2026 Problem Statement 26066)
Prototype Launcher

Starts the judge-facing FastAPI application on http://localhost:8000
and automatically opens the default web browser.
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser(url: str, delay: float = 1.2):
    time.sleep(delay)
    print(f"\n[OceanEmbed] Opening working prototype in browser: {url}")
    webbrowser.open(url)

def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 70)
    print("🌊 OceanEmbed: Judge-Facing Working Prototype")
    print("   SIH 2026 | Problem Statement 26066")
    print("=" * 70)
    print(f"Starting server on {url} ...")

    # Launch browser after a short delay
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Start FastAPI / Uvicorn server
    uvicorn.run("dashboard.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
