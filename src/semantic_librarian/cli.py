"""The ``sl`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantic_librarian import __version__
from semantic_librarian.embed.registry import EMBEDDERS
from semantic_librarian.ingest import FORMATS, FieldMapping
from semantic_librarian.library import Hit, Library, LibraryError
from semantic_librarian.search.similarity import QUERY_MODES
from semantic_librarian.text.chunk import DEFAULT_CHUNK_WORDS


def _split_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_params(items: Sequence[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        params[key.strip().replace("-", "_")] = _parse_value(value.strip())
    return params


def _print_hits(hits: Sequence[Hit], width: int = 90) -> None:
    for h in hits:
        if h.document is not None:
            title = " ".join(h.document.title().split())
            year = h.document.factors.get("year", [h.document.meta.get("year", "")])[0]
            line = f"{h.rank:>4}  {h.score:7.4f}  {title}"
            if year:
                line += f" ({year})"
            print(line[:width])
            if h.chunk is None:
                print(f"{'':15}{h.label}")
                continue
            first, last = h.chunk.meta.get("page"), h.chunk.meta.get("page_end")
            pages = f"p. {first}" if first == last else f"pp. {first}-{last}"
            print(f"{'':15}{h.label}  {pages}" if first else f"{'':15}{h.label}")
            passage = textwrap.shorten(h.chunk.text, 2 * (width - 15), placeholder=" ...")
            for row in textwrap.wrap(passage, width - 15):
                print(f"{'':15}{row}")
        else:
            print(f"{h.rank:>4}  {h.score:7.4f}  {h.label}"[:width])


def cmd_init(args: argparse.Namespace) -> int:
    settings: dict[str, Any] = {"text_mode": args.text_mode, "embedder": args.embedder}
    lib = Library.create(args.library, name=args.name, **settings)
    print(f"created library {lib.config.name!r} at {lib.root}")
    print("next: sl add", lib.root, "FILES...")
    return 0


def _expand(paths: Sequence[Path]) -> list[Path]:
    """Replace each directory by the files inside it (at any depth) that can be ingested."""
    known = {suffix for suffixes in FORMATS.values() for suffix in suffixes}
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in known)
        else:
            files.append(path)
    return files


def cmd_add(args: argparse.Namespace) -> int:
    with Library.open(args.library) as lib:
        mapping = None
        if args.text or args.factor is not None or args.id:
            mapping = FieldMapping.from_options(
                text=_split_list(args.text),
                factors=args.factor,
                id=args.id,
            )
        failed = 0
        for path in _expand(args.files):
            try:
                report = lib.add_file(
                    path,
                    fmt=args.format,
                    mapping=mapping,
                    split=args.split,
                    force=args.force,
                    chunk_words=args.chunk_words,
                )
            except ValueError as exc:
                # One unreadable file should not stop a whole folder from being added.
                print(f"error: {path}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if report.skipped:
                print(f"unchanged, skipped: {path}")
            else:
                print(f"added {report.added:>6} documents from {path} ({report.format})")
        n = lib.store.count_documents()
        chunks = lib.store.count_chunks()
        held = f"{n} documents" + (f" ({chunks} chunks)" if chunks else "")
        print(f"library now holds {held}; run `sl build {args.library}` to (re)embed")
        if failed:
            print(f"{failed} file(s) could not be added", file=sys.stderr)
    return 2 if failed else 0


def cmd_build(args: argparse.Namespace) -> int:
    with Library.open(args.library) as lib:
        params = _parse_params(args.param)
        if args.dim is not None:
            params["dim"] = args.dim
        if args.seed is not None:
            params["seed"] = args.seed
        if args.text_mode:
            lib.config.text_mode = args.text_mode
        report = lib.build(
            embedder=args.embedder,
            params=params or None,
            log=None if args.quiet else (lambda m: print(f"  {m}")),
        )
        spaces = ", ".join(f"{k} ({v})" for k, v in report.spaces.items())
        print(f"built with {report.embedder}: {spaces}")
    return 0


def _year_filter(spec: str | None) -> Any:
    if not spec:
        return None
    lo_s, _, hi_s = spec.partition("-")
    lo = int(lo_s) if lo_s else -(10**9)
    hi = int(hi_s) if hi_s else 10**9

    def where(doc: Any) -> bool:
        raw = doc.factors.get("year", [doc.meta.get("year")])[0]
        try:
            return lo <= int(str(raw)[:4]) <= hi
        except (TypeError, ValueError):
            return False

    return where


def cmd_search(args: argparse.Namespace) -> int:
    with Library.open(args.library) as lib:
        if lib.is_stale:
            print("warning: documents were added since the last build", file=sys.stderr)
        result = lib.search(
            args.query,
            mode=args.mode,
            space=args.space,
            k=args.k,
            where=_year_filter(args.years),
            method=args.method,
            lexical_weight=args.lexical_weight,
            include_stopwords=args.stopwords,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if result.unknown:
            print(f"ignored (not in vocabulary): {', '.join(result.unknown)}")
        if not result.terms:
            print("no query words are in the library's vocabulary")
            return 1
        style = args.mode if args.method == "semantic" else args.method
        print(f"{style} search in {args.space} space for: {' '.join(result.terms)}\n")
        _print_hits(result.hits)
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    with Library.open(args.library) as lib:
        hits = lib.similar(
            args.item,
            space=args.space,
            target=args.target,
            k=args.k,
            where=_year_filter(args.years),
            include_stopwords=args.stopwords,
        )
        if args.json:
            print(json.dumps([h.to_dict() for h in hits], indent=2, ensure_ascii=False))
        else:
            _print_hits(hits)
    return 0


def _built_models(lib: Library, cfg: Any) -> dict[str, Any]:
    """The library's built model, if it matches an evaluated method's settings."""
    from semantic_librarian.embed.registry import make_embedder

    if not lib.is_built or lib.is_stale:
        return {}
    name = lib.config.embedder
    if name not in [*cfg.methods, cfg.reference]:
        return {}

    def comparable(config: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in config.items() if k != "workers"}

    wanted = make_embedder(name, **cfg.params.get(name, {})).config()
    built = lib.build_info["embedder"]
    return {name: lib.embedder} if comparable(wanted) == comparable(built) else {}


def cmd_evaluate(args: argparse.Namespace) -> int:
    from semantic_librarian.eval import EvalConfig, evaluate

    params: dict[str, dict[str, Any]] = {}
    methods = _split_list(args.methods) or []
    if args.dim is not None:
        for m in [*methods, args.reference]:
            if m not in ("wordmatch", "bm25", "hybrid"):
                params.setdefault(m, {})["dim"] = args.dim
    for key, value in _parse_params(args.param).items():
        method, dot, name = key.partition(".")
        if not dot:
            raise SystemExit(f"--param for evaluate expects METHOD.KEY=VALUE, got {key!r}")
        params.setdefault(method, {})[name] = value
    cfg = EvalConfig(
        methods=methods,
        percents=[int(p) for p in _split_list(args.percent) or []],
        n_targets=args.n,
        simulations=[int(x) for x in _split_list(args.simulations) or []],
        seed=args.seed,
        reference=args.reference,
        params=params,
        min_tokens=args.min_tokens,
        lexical_weight=args.lexical_weight,
        associates=args.associates,
    )
    with Library.open(args.library) as lib:
        say = None if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr))
        results = evaluate(lib.corpus(), cfg, log=say, prefit=_built_models(lib, cfg))
    print(results.format_text())
    if args.out:
        Path(args.out).write_text(results.to_json())
        print(f"\nfull results written to {args.out}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    with Library.open(args.library) as lib:
        print(f"library   {lib.config.name}  ({lib.root})")
        print(f"documents {lib.store.count_documents()}")
        if lib.store.count_chunks():
            print(f"chunks    {lib.store.count_chunks()}")
        for s in lib.store.sources():
            print(f"  source  {s['documents']:>6}  {s['kind']:<7} {s['path']}")
        print(
            f"text mode {lib.config.text_mode}; embedder {lib.config.embedder} "
            f"{lib.config.embedder_params or ''}"
        )
        if lib.is_built:
            info = lib.build_info
            print(
                f"built     {info['created_at']} in {info['seconds']:.1f}s, "
                f"vocabulary {info['vocab_size']}"
            )
            for name, n in info["spaces"].items():
                print(f"  space   {name:<12} {n:>7} items")
            if lib.is_stale:
                print("stale     documents changed since the last build; run `sl build`")
        else:
            print("not built yet; run `sl build`")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from semantic_librarian.export import export_library

    with Library.open(args.library) as lib:
        say = None if args.quiet else (lambda m: print(f"  {m}"))
        report = export_library(lib, args.out, precision=args.precision, log=say)
    print(f"exported to {report.path} ({report.megabytes:.1f} MB)")
    print(f"view it with: sl serve {args.library}, or put the folder on any static web host")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from semantic_librarian.export import export_library, serve
    from semantic_librarian.export.bundle import app_version

    with Library.open(args.library) as lib:
        out = lib.root / "web"
        manifest_file = out / "data" / "manifest.json"
        manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
        # reuse the last export unless the library was rebuilt or the settings changed
        current = (
            manifest.get("built_at") == lib.build_info["created_at"]
            and manifest.get("spaces", [{}])[0].get("dtype") == args.precision
            and manifest.get("app") == app_version()
        )
        if args.fresh or not current:
            report = export_library(lib, out, precision=args.precision)
            print(f"exported to {report.path} ({report.megabytes:.1f} MB)")
    serve(out, port=args.port, open_browser=args.open)
    return 0


def cmd_embedders(args: argparse.Namespace) -> int:
    for name, desc in EMBEDDERS.items():
        print(f"{name:<14} {desc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sl", description="Semantic Librarian: vector-space search over your documents."
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="create a new library directory")
    s.add_argument("library", type=Path)
    s.add_argument("--name")
    s.add_argument("--text-mode", choices=["unicode", "legacy"], default="unicode")
    s.add_argument("--embedder", choices=list(EMBEDDERS), default="beagle")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("add", help="ingest files or folders (json, jsonl, csv, bib, txt/md, pdf)")
    s.add_argument("library", type=Path)
    s.add_argument("files", nargs="+", type=Path, help="files, or folders to search for them")
    s.add_argument("--format", choices=list(FORMATS))
    s.add_argument("--text", help="comma-separated fields to embed (json/csv)")
    s.add_argument(
        "--factor",
        action="append",
        help="factor spec NAME, NAME=KEY or NAME=KEY:SEP (repeatable; json/csv)",
    )
    s.add_argument("--id", help="field holding the document id (json/csv)")
    s.add_argument(
        "--split",
        choices=["document", "paragraph", "sentence"],
        default="document",
        help="for text files: one document per file, paragraph or sentence",
    )
    s.add_argument(
        "--chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help=f"for PDFs: approximate words per chunk (default {DEFAULT_CHUNK_WORDS})",
    )
    s.add_argument("--force", action="store_true", help="re-ingest unchanged files")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("build", help="embed all documents and write the vector spaces")
    s.add_argument("library", type=Path)
    s.add_argument("--embedder", choices=list(EMBEDDERS))
    s.add_argument("--dim", type=int)
    s.add_argument("--seed", type=int)
    s.add_argument("--text-mode", choices=["unicode", "legacy"])
    s.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="embedder parameter, e.g. max_ngram=5 (repeatable)",
    )
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="semantic search with a free-text query")
    s.add_argument("library", type=Path)
    s.add_argument("query")
    s.add_argument("--mode", choices=QUERY_MODES, default="compound")
    s.add_argument(
        "--space",
        default="document",
        help="space to rank: document, word, chunk (passages of PDFs), or a factor such as author",
    )
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--years", help="year range for documents, e.g. 1990-2010 or 2000-")
    s.add_argument(
        "--method",
        choices=["semantic", "lexical", "hybrid"],
        default="semantic",
        help="vector-space, keyword (BM25), or both fused (default: semantic)",
    )
    s.add_argument(
        "--lexical-weight",
        type=float,
        default=0.5,
        help="share of the keyword ranking in hybrid search, 0-1 (default 0.5)",
    )
    s.add_argument("--stopwords", action="store_true", help="show stop words in word space")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("similar", help="items closest to an existing item")
    s.add_argument("library", type=Path)
    s.add_argument("item", help="document id, word, or factor level (e.g. an author name)")
    s.add_argument("--space", default="document", help="space the item belongs to")
    s.add_argument("--target", help="space to rank (default: same as --space)")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--years", help="year range when the target is the document space")
    s.add_argument("--stopwords", action="store_true", help="show stop words in word space")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_similar)

    s = sub.add_parser(
        "evaluate", help="compare retrieval methods with the paper's simulations 1-3"
    )
    s.add_argument("library", type=Path)
    s.add_argument(
        "--methods",
        default="beagle,beagle-rp,random,wordmatch,bm25,hybrid",
        help="comma-separated embedders and/or wordmatch, bm25, hybrid",
    )
    s.add_argument("--percent", default="5,10,25,50,100", help="query sizes, %% of a document")
    s.add_argument("-n", type=int, default=200, help="target documents (default 200)")
    s.add_argument("--simulations", default="1,2,3")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--reference", default="beagle", help="model supplying associates")
    s.add_argument("--dim", type=int, help="vector dimension for every embedder")
    s.add_argument(
        "--param",
        action="append",
        metavar="METHOD.KEY=VALUE",
        help="embedder parameter, e.g. beagle.max_ngram=3 (repeatable)",
    )
    s.add_argument("--min-tokens", type=int, default=10)
    s.add_argument("--lexical-weight", type=float, default=0.5)
    s.add_argument(
        "--associates",
        choices=["nearest", "outside-document"],
        default="nearest",
        help="simulation 2: nearest neighbor (paper), or nearest not in the target document",
    )
    s.add_argument("--out", help="write all results as JSON to this file")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("export", help="write the library as a static web app")
    s.add_argument("library", type=Path)
    s.add_argument("out", type=Path, help="folder to write (new, empty, or an earlier export)")
    s.add_argument(
        "--precision",
        choices=["int8", "float32"],
        default="int8",
        help="vector storage: int8 (compact, default) or float32 (exact, 4x larger)",
    )
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("serve", help="search the library in a web browser on this computer")
    s.add_argument("library", type=Path)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--open", action="store_true", help="open the browser")
    s.add_argument("--precision", choices=["int8", "float32"], default="int8")
    s.add_argument("--fresh", action="store_true", help="export again even if up to date")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("info", help="summarize a library")
    s.add_argument("library", type=Path)
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("embedders", help="list available embedders")
    s.set_defaults(func=cmd_embedders)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (LibraryError, KeyError, ValueError, NotImplementedError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
