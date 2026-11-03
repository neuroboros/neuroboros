"""Pre-render step: run ``quartodoc build`` only when its inputs have changed.

quartodoc rewrites ``reference/index.qmd``, ``reference/_sidebar.yml``, and
``objects.json`` on every run, even when nothing changed. ``quarto preview``
watches those files, so running it unconditionally makes every render trigger
another one, and the reload it sends to the browser can cancel a navigation in
progress. The fingerprint of the package source and the project configuration
is stored next to the generated pages, and quartodoc runs only when it differs.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
SRC = DOCS.parent / "src" / "neuroboros"
REFERENCE = DOCS / "reference"
STAMP = REFERENCE / ".build-stamp"


def fingerprint():
    h = hashlib.sha256()
    for fn in sorted([DOCS / "_quarto.yml", *SRC.rglob("*.py")]):
        h.update(str(fn.relative_to(DOCS.parent)).encode())
        h.update(fn.read_bytes())
    return h.hexdigest()


def main():
    fp = fingerprint()
    if (REFERENCE / "index.qmd").exists() and STAMP.exists() and STAMP.read_text().strip() == fp:
        print("quartodoc: reference pages are up to date")
        return
    subprocess.run([sys.executable, "-m", "quartodoc", "build"], cwd=DOCS, check=True)
    REFERENCE.mkdir(exist_ok=True)
    STAMP.write_text(fp + "\n")


if __name__ == "__main__":
    main()
