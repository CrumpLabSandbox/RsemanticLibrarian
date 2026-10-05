"""A library of research projects funded by the U.S. National Science Foundation.

NSF publishes every award it makes, with an abstract written for the public, and states
that this data is in the public domain. One fiscal year is about 11,000 awards. This
script builds a library in which words, projects, investigators, institutions, states,
funding programs, divisions and directorates all share one space.

    python examples/nsf_awards.py [LIBRARY_FOLDER] [YEAR ...]

Fiscal year 2024 comes with the repository, already prepared, as
``nsf-data/nsf-awards-2024.jsonl``. Any other year is downloaded from NSF (50-160 MB)
and prepared the same way. Awards that share an abstract, as the parts of a
collaborative project do, become one document that carries the investigators and
institutions of all its parts. Of each award only the title, the abstract, the names of
people and organizations, the amount, the type and the start date are kept; contact
details in NSF's files are not.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from semantic_librarian import Document, Library
from semantic_librarian.ingest import FieldMapping

FILE_LIST = "https://api.nsf.gov/services/v2/s3/list-files"
AWARD_PAGE = "https://www.nsf.gov/awardsearch/showAward?AWD_ID="
DATA = Path("nsf-data")
# The sentence NSF appends to every abstract; it says nothing about the project.
BOILERPLATE = "This award reflects NSF's statutory mission"
PREFIX = "Collaborative Research:"
FACTORS = ("investigator", "institution", "state", "program", "division", "directorate", "year")


def download(year: int, folder: Path = DATA) -> Path:
    """Fetch one fiscal year's awards (a zip of JSON files) unless it is already here."""
    target = folder / f"{year}.zip"
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(FILE_LIST, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request) as response:
        files = json.load(response)["files"]
    entry = next((f for f in files if f["fileName"] == f"{year}.zip"), None)
    if entry is None:
        raise SystemExit(f"NSF lists no file for fiscal year {year}")
    print(f"downloading {entry['fileName']} ({entry['size'] / 1e6:.0f} MB) from NSF")
    urllib.request.urlretrieve(entry["downloadUrl"], target)
    return target


def read_awards(archive: Path) -> Iterator[dict[str, Any]]:
    """The award records in a downloaded zip, one JSON file each."""
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if name.endswith(".json"):
                yield json.loads(z.read(name))


def clean_abstract(text: str | None) -> str:
    text = (text or "").split(BOILERPLATE)[0]
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def clean_title(title: str | None) -> str:
    title = " ".join((title or "").split())
    return title[len(PREFIX) :].strip() if title.startswith(PREFIX) else title


def to_documents(awards: Iterable[dict[str, Any]], year: int) -> list[Document]:
    """One document per project; awards with the same abstract are merged."""
    projects: dict[str, Document] = {}
    for award in awards:
        abstract = clean_abstract(award.get("awd_abstract_narration"))
        if not abstract:
            continue
        institution = award.get("inst") or {}
        factors = {
            "investigator": [p.get("pi_full_name", "") for p in award.get("pi") or []],
            "institution": [institution.get("inst_name", "")],
            "state": [institution.get("inst_state_name", "")],
            "program": [p.get("pgm_ele_name", "") for p in award.get("pgm_ele") or []],
            "division": [award.get("org_div_long_name", "")],
            "directorate": [award.get("org_dir_long_name", "")],
            "year": [str(year)],
        }
        doc = projects.get(abstract)
        if doc is None:
            projects[abstract] = Document(
                id=str(award["awd_id"]),
                fields={"title": clean_title(award.get("awd_titl_txt")), "abstract": abstract},
                meta={
                    "award": AWARD_PAGE + str(award["awd_id"]),
                    "type": award.get("awd_istr_txt"),
                    "starts": award.get("awd_eff_date"),
                    "amount": award.get("awd_amount") or 0,
                    "awards": 1,
                },
                factors=factors,
            )
            continue
        doc.meta["awards"] += 1
        doc.meta["amount"] += award.get("awd_amount") or 0
        for name, levels in factors.items():
            known = doc.factors.setdefault(name, [])
            known += [level for level in levels if level.strip() and level not in known]
    return list(projects.values())


def prepared(year: int, folder: Path = DATA) -> Path:
    """The year as a JSON Lines file, one project per line, made on first use."""
    target = folder / f"nsf-awards-{year}.jsonl"
    if not target.exists():
        docs = to_documents(read_awards(download(year, folder)), year)
        with target.open("w", encoding="utf-8") as out:
            for doc in docs:
                record = {"id": doc.id, **doc.fields, **doc.factors, **doc.meta}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def main(folder: str = "libraries/nsf", *years: str) -> None:
    if Path(folder, "config.toml").exists():
        lib = Library.open(folder)
    else:
        lib = Library.create(folder)
        # the same as: sl add LIBRARY FILE --text title,abstract --id id --factor investigator ...
        mapping = FieldMapping.from_options(text=["title", "abstract"], factors=FACTORS, id="id")
        for year in [int(y) for y in years] or [2024]:
            report = lib.add_file(prepared(year), mapping=mapping)
            print(f"fiscal year {year}: {report.added} projects")
        lib.build(log=lambda message: print(" ", message))
    print({name: len(space) for name, space in lib.spaces.items()})

    query = "sea ice and polar ecosystems"
    for space in ("document", "program", "institution", "word"):
        print(f"\n{space}s closest to: {query}")
        for hit in lib.search(query, space=space, k=5).hits:
            label = hit.document.title() if hit.document else hit.label
            print(f"{hit.rank:>3}  {hit.score:.3f}  {label[:80]}")
    lib.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
