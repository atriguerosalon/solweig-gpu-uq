"""Make result files safe to publish: absolute local paths become repo-relative or home-relative.

Result JSON files record the inputs they were made from (met files, preprocessed tiles), as
absolute paths on the machine that ran them. This rewrites, in place, the given files:

    <repo>/...                 ->  ...            (repo-relative)
    /mnt/c/Users/<u>/...       ->  ~/...          (Windows home, seen from WSL)
    /home/<u>/...              ->  ~/...
    C:\\Users\\<u>\\... (any slash) ->  ~/...

Only text files are touched; the number of replacements per file is printed. Run before committing
new result files.

Usage:
    python scripts/strip_local_paths.py figures/*.json figures/*/*.json
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def strip(text: str) -> tuple[str, int]:
    n = 0
    repo_wsl = "/mnt/" + REPO.as_posix()[0].lower() + REPO.as_posix()[2:]  # C:/x/y -> /mnt/c/x/y
    for prefix in {repo_wsl + "/", REPO.as_posix() + "/", str(REPO).replace("\\", "\\\\") + "\\\\"}:
        n += text.count(prefix)
        text = text.replace(prefix, "")
    for pat in (r"/mnt/[a-z]/Users/[^/\"]+", r"/home/[^/\"]+", r"[A-Za-z]:(?:\\\\|/|\\)Users(?:\\\\|/|\\)[^/\\\"]+"):
        text, k = re.subn(pat, "~", text)
        n += k
    return text, n


def main() -> None:
    for name in sys.argv[1:]:
        p = Path(name)
        t = p.read_text(encoding="utf-8")
        new, n = strip(t)
        if n:
            p.write_bytes(new.encode("utf-8"))
            print(f"{p}: {n} paths")


if __name__ == "__main__":
    main()
