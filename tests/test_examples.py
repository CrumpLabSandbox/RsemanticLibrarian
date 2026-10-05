"""The example scripts behind the tutorials keep running."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"

STORY = """A short book

CHAPTER I. The Garden
Alice walked into the garden. The roses in the garden were red and white.
A gardener painted the white roses red. "Why paint the roses?" said Alice.

CHAPTER II. The Tea Party
The Hatter poured tea for the Hare. The Hare dipped his watch in the tea.
Alice sat at the tea table. The Hatter asked Alice a riddle about a raven.
"""


def run(name: str, *args: object) -> None:
    runpy.run_path(str(EXAMPLES / name))["main"](*[str(a) for a in args])


def test_abstracts_example(tmp_path, capsys):
    run("abstracts.py", tmp_path / "lib")
    out = capsys.readouterr().out
    assert "query words used: ['president', 'psychology']" in out
    assert "authors closest to:" in out and "cluster" in out
    run("abstracts.py", tmp_path / "lib")  # a second run reuses the library
    assert capsys.readouterr().out == out


def test_bibliography_example(tmp_path, capsys):
    run("bibliography.py", EXAMPLES.parent / "tests" / "fixtures" / "crump.bib", tmp_path / "lib")
    out = capsys.readouterr().out
    assert "'document': 39" in out and "typists" in out


def test_any_text_example(tmp_path, capsys, monkeypatch):
    story = tmp_path / "story.txt"
    story.write_text(STORY)
    run("any_text.py", story, tmp_path / "lib", "tea with the hatter")
    out = capsys.readouterr().out
    assert "'document': 9" in out and "'chapter': 2" in out
    assert "[CHAPTER II. The Tea Party]" in out.split("chapters closest to")[0]
    # the tea-party chapter is the chapter closest to a query about tea
    assert (
        out.split("chapters closest to: tea with the hatter")[1]
        .split("\n")[1]
        .endswith("CHAPTER II. The Tea Party")
    )
    plain = tmp_path / "plain.txt"
    plain.write_text("One sentence here. Another sentence there.")
    run("any_text.py", plain, tmp_path / "lib2")
    assert "'chapter': 1" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["any_text.py"])
    with pytest.raises(SystemExit):
        runpy.run_path(str(EXAMPLES / "any_text.py"), run_name="__main__")


def test_nsf_awards_example(tmp_path):
    """Award records become project documents; the download itself is not tested."""
    import json
    import zipfile

    nsf = runpy.run_path(str(EXAMPLES / "nsf_awards.py"))
    closing = " This award reflects NSF's statutory mission and has been deemed worthy."

    def award(number, title, abstract, person, place, state="Alaska"):
        return {
            "awd_id": str(number),
            "awd_titl_txt": title,
            "awd_abstract_narration": abstract,
            "awd_amount": 100.0,
            "awd_istr_txt": "Standard Grant",
            "awd_eff_date": "2024-01-01",
            "org_dir_long_name": "Directorate for Geosciences",
            "org_div_long_name": "Office of Polar Programs",
            "pi": [{"pi_full_name": person}],
            "inst": {"inst_name": place, "inst_state_name": state},
            "pgm_ele": [{"pgm_ele_name": "Arctic Natural Sciences"}],
        }

    shared = "Sea ice is thinning.\r\n\r\n  The team will measure it."
    archive = tmp_path / "2024.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "1.json",
            json.dumps(award(1, "Collaborative Research: Ice", shared + closing, "A B", "U One")),
        )
        z.writestr(
            "2.json",
            json.dumps(award(2, "Collaborative Research: Ice", shared + closing, "C D", "U Two")),
        )
        z.writestr(
            "3.json", json.dumps(award(3, "Whales", "Whales sing." + closing, "E F", "U One"))
        )
        z.writestr("4.json", json.dumps(award(4, "No abstract", None, "G H", "U Three")))
        z.writestr("readme.txt", "not an award")
    assert nsf["download"](2024, tmp_path) == archive  # already there, so nothing is fetched

    ice, whales = nsf["to_documents"](nsf["read_awards"](archive), 2024)
    assert ice.id == "1" and ice.fields["title"] == "Ice"
    assert ice.fields["abstract"] == "Sea ice is thinning.\nThe team will measure it."
    assert ice.factors["investigator"] == ["A B", "C D"]
    assert ice.factors["institution"] == ["U One", "U Two"]
    assert ice.factors["state"] == ["Alaska"] and ice.factors["year"] == ["2024"]
    assert ice.meta["awards"] == 2 and ice.meta["amount"] == 200.0
    assert ice.meta["award"].endswith("AWD_ID=1")
    assert whales.fields == {"title": "Whales", "abstract": "Whales sing."}

    # the prepared file holds the same projects, and nothing but the fields listed here
    lines = nsf["prepared"](2024, tmp_path).read_text().splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 2 and first["investigator"] == ["A B", "C D"]
    assert set(first) == {"id", "title", "abstract", "award", "type", "starts", "amount"} | {
        "awards",
        *nsf["FACTORS"],
    }
