"""
Pre-deploy checks for the static bundle in `web/`.

There is no build step, so nothing else would catch a renamed asset or a stale
data file before it reaches production. Run it locally or let CI run it:

    python scripts/check_web_bundle.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

# Long dashes read as machine-written and are banned from anything a reader
# sees. Hyphens and the true minus sign U+2212 are fine.
# Written as escapes so this file does not trip its own check.
BANNED_CHARACTERS = {
    chr(0x2014): "em dash",
    chr(0x2013): "en dash",
}

PROSE_FILES = [
    ROOT / "README.md",
    WEB / "index.html",
    WEB / "styles.css",
    WEB / "app.js",
    ROOT / "Home.py",
    ROOT / "case_service.py",
    ROOT / "data_service.py",
    *sorted((ROOT / "pages").glob("*.py")),
    *sorted((ROOT / "scripts").glob("*.py")),
]

REQUIRED_FILES = [
    WEB / "index.html",
    WEB / "styles.css",
    WEB / "app.js",
    WEB / "data" / "co2.json",
    WEB / "assets" / "plant.webp",
    WEB / "fonts" / "inter-latin.woff2",
    WEB / "fonts" / "space-grotesk-latin.woff2",
    WEB / "fonts" / "jetbrains-mono-latin.woff2",
]

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


def check_required_files() -> None:
    for path in REQUIRED_FILES:
        if not path.exists():
            fail(f"missing required file: {path.relative_to(ROOT)}")


def check_local_references() -> None:
    """Every relative href/src in index.html must resolve to a real file."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:href|src)="([^"]+)"', html)

    for reference in references:
        if reference.startswith(("http://", "https://", "data:", "mailto:", "#", "//")):
            continue
        target = (WEB / reference.split("?")[0].split("#")[0]).resolve()
        if not target.exists():
            fail(f"index.html references a missing file: {reference}")


def check_data_payload() -> None:
    path = WEB / "data" / "co2.json"
    if not path.exists():
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"data/co2.json is not valid JSON: {error}")
        return

    for key in ("meta", "national", "spread", "cases"):
        if key not in payload:
            fail(f"data/co2.json is missing the '{key}' section")

    cases = payload.get("cases", {})
    for code in ("WY", "ND", "AK"):
        if code not in cases:
            fail(f"data/co2.json has no case study for {code}")

    spread = payload.get("spread", {}).get("values", [])
    if len(spread) != 50:
        fail(f"data/co2.json spread covers {len(spread)} states, expected 50")


def check_reader_facing_dashes() -> None:
    for path in PROSE_FILES:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for character, name in BANNED_CHARACTERS.items():
                if character in line:
                    column = line.index(character) + 1
                    fail(
                        f"{path.relative_to(ROOT)}:{number}:{column} contains an "
                        f"{name}; use a hyphen instead"
                    )


def check_no_placeholders() -> None:
    pattern = re.compile(r"\b(TODO|FIXME|XXX|lorem ipsum)\b", re.IGNORECASE)
    for path in [WEB / "index.html", WEB / "app.js", WEB / "styles.css"]:
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = pattern.search(line)
            if match:
                fail(f"{path.relative_to(ROOT)}:{number} left a {match.group(0)} behind")


def main() -> int:
    check_required_files()
    check_local_references()
    check_data_payload()
    check_reader_facing_dashes()
    check_no_placeholders()

    if failures:
        print(f"{len(failures)} problem(s) found:\n")
        for problem in failures:
            print(f"  {problem}")
        return 1

    total = sum(path.stat().st_size for path in WEB.rglob("*") if path.is_file())
    print(f"web bundle looks good ({total / 1024:.0f} KB across {len(REQUIRED_FILES)} required files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
