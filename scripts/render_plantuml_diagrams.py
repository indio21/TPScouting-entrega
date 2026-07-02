from __future__ import annotations

import sys
import urllib.request
import zlib
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "diagramas" / "plantuml"
OUTPUT_DIR = ROOT / "docs" / "diagramas" / "export"
PLANTUML_SERVER = "https://www.plantuml.com/plantuml"

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def encode_plantuml(source: str) -> str:
    compressed = zlib.compress(source.encode("utf-8"))[2:-4]
    encoded = []
    for index in range(0, len(compressed), 3):
        chunk = compressed[index : index + 3]
        if len(chunk) == 1:
            b1, b2, b3 = chunk[0], 0, 0
        elif len(chunk) == 2:
            b1, b2, b3 = chunk[0], chunk[1], 0
        else:
            b1, b2, b3 = chunk
        c1 = b1 >> 2
        c2 = ((b1 & 0x3) << 4) | (b2 >> 4)
        c3 = ((b2 & 0xF) << 2) | (b3 >> 6)
        c4 = b3 & 0x3F
        encoded.extend(ALPHABET[value] for value in (c1, c2, c3, c4))
    return "".join(encoded)


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "TPScouting-docs/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    target.write_bytes(data)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SOURCE_DIR.glob("*.puml"))
    if not sources:
        raise SystemExit(f"No se encontraron archivos .puml en {SOURCE_DIR}")
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        encoded = encode_plantuml(source)
        png_path = OUTPUT_DIR / f"{source_path.stem}.png"
        svg_path = OUTPUT_DIR / f"{source_path.stem}.svg"
        download(f"{PLANTUML_SERVER}/png/{encoded}", png_path)
        download(f"{PLANTUML_SERVER}/svg/{encoded}", svg_path)
        print(f"Generado: {png_path.relative_to(ROOT)}")
        print(f"Generado: {svg_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
