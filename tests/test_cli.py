from __future__ import annotations

import json

from semantic_librarian.cli import main


def test_cli_workflow(tmp_path, crump_bib, capsys):
    lib = str(tmp_path / "lib")
    assert main(["init", lib]) == 0
    assert main(["add", lib, str(crump_bib)]) == 0
    assert main(["add", lib, str(crump_bib)]) == 0
    assert "unchanged" in capsys.readouterr().out
    assert main(["build", lib, "--dim", "128", "--param", "max_ngram=3", "-q"]) == 0
    assert "built with beagle" in capsys.readouterr().out

    assert main(["search", lib, "keyboard typing", "-k", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["terms"] == ["keyboard", "typing"] and len(data["hits"]) == 2

    assert main(["search", lib, "attention", "--mode", "or", "--years", "2015-"]) == 0
    assert "or search in document space" in capsys.readouterr().out
    assert main(["search", lib, "keyboard", "--method", "lexical", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "lexical search" in out and "Keyboard" in out
    assert main(["search", lib, "keyboard typing", "--method", "hybrid", "--json"]) == 0
    hybrid = json.loads(capsys.readouterr().out)
    assert hybrid["method"] == "hybrid" and "components" in hybrid["hits"][0]
    assert main(["search", lib, "zebra"]) == 1
    assert main(["similar", lib, "Logan, Gordon D.", "--space", "author", "-k", "2"]) == 0
    assert "Logan, Gordon D." in capsys.readouterr().out
    assert main(["info", lib]) == 0
    assert "space   author" in capsys.readouterr().out
    assert main(["embedders"]) == 0


def test_cli_evaluate(tmp_path, crump_bib, capsys):
    lib = str(tmp_path / "lib")
    main(["init", lib])
    main(["add", lib, str(crump_bib)])
    capsys.readouterr()
    out_file = tmp_path / "eval.json"
    args = [
        "evaluate",
        lib,
        "--methods",
        "beagle,random,bm25",
        "--percent",
        "10,100",
        "-n",
        "8",
        "--dim",
        "64",
        "--param",
        "beagle.max_ngram=3",
        "--out",
        str(out_file),
        "-q",
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Simulation 1" in out and "Simulation 2" in out and "Simulation 3" in out
    data = json.loads(out_file.read_text())
    assert data["config"]["params"]["beagle"] == {"dim": 64, "max_ngram": 3}
    assert {r["method"] for r in data["summary"]["simulation_1"]} == {"beagle", "random", "bm25"}

    # a built library's model is reused when its settings match
    main(["build", lib, "--dim", "64", "--param", "max_ngram=3", "-q"])
    capsys.readouterr()
    args = [
        "evaluate",
        lib,
        "--methods",
        "beagle,bm25",
        "--percent",
        "50",
        "-n",
        "4",
        "--dim",
        "64",
        "--param",
        "beagle.max_ngram=3",
        "--associates",
        "outside-document",
    ]
    assert main(args) == 0
    assert "using the library's built beagle vectors" in capsys.readouterr().err


def test_cli_errors(tmp_path, capsys):
    assert main(["info", str(tmp_path / "missing")]) == 2
    assert "not a library" in capsys.readouterr().err
    lib = str(tmp_path / "lib")
    main(["init", lib])
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert main(["add", lib, str(pdf)]) == 2
    assert "cannot read paper.pdf as a PDF" in capsys.readouterr().err
