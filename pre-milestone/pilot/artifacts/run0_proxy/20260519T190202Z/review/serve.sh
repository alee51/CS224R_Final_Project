#!/usr/bin/env sh
# Serve dashboard so KaTeX CDN and fonts load reliably (file:// often blocks or breaks math).
cd "$(dirname "$0")"
echo "Open http://localhost:8765 in your browser"
exec python3 -m http.server 8765
