"""Build the public website on this machine: the documentation plus a live demo.

    python scripts/publish_site.py            # build into site/ and stop
    python scripts/publish_site.py --push     # also publish site/ to the gh-pages branch

The site is the mkdocs documentation with the NSF awards library exported under
``demo/``, where the visitor's browser does the searching. Nothing here needs a server,
so GitHub Pages can host it: set Pages to deploy from the ``gh-pages`` branch.

Needs the ``docs`` extra. The NSF library is built first if it does not exist (about
three minutes). ``--push`` replaces the gh-pages branch with a single commit, so the
branch does not grow with every publish.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from semantic_librarian import Library
from semantic_librarian.export import export_library

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
LIBRARY = ROOT / "libraries" / "nsf"
# Words rarer than this are left out of the demo to shorten the first download.
MIN_COUNT = 5


def run(*command: str) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--push", action="store_true", help="publish to the gh-pages branch")
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()

    if not (LIBRARY / "vectors" / "build.json").exists():
        run(sys.executable, "examples/nsf_awards.py", str(LIBRARY))
    run(sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(SITE))
    with Library.open(LIBRARY) as lib:
        report = export_library(lib, SITE / "demo", min_count=MIN_COUNT, log=print)
    largest = max((f for f in SITE.rglob("*") if f.is_file()), key=lambda f: f.stat().st_size)
    total = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6
    print(f"site: {total:.0f} MB in {SITE}; demo {report.megabytes:.0f} MB")
    print(f"largest file: {largest.relative_to(SITE)} ({largest.stat().st_size / 1e6:.0f} MB)")

    if args.push:
        run(
            sys.executable,
            "-m",
            "ghp_import",
            "--no-jekyll",
            "--no-history",
            "--force",
            "--push",
            "--remote",
            args.remote,
            "--message",
            "Publish the documentation and the NSF demo",
            str(SITE),
        )
        print("published. In the repository settings, set Pages to the gh-pages branch.")
    else:
        print(f"preview: python -m http.server 8004 --directory {SITE.relative_to(ROOT)}")
        print("publish: python scripts/publish_site.py --push")


if __name__ == "__main__":
    main()
