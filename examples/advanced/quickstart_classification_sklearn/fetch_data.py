"""Download, verify and prepare the quickstart_classification data.

Usage (any Python 3.12+; standard library only)::

    python fetch_data.py --output_dir DIR                 # download from Zenodo
    python fetch_data.py --output_dir DIR --source_dir S  # use files already in S

Reads ``data_source.json`` next to this script. Every file listed there is
checked for size and md5 before use. The Zenodo record holds four shards with
the header ``sample_id,x0,x1,x2,x3,label``; this script writes the layout that
``task/task.yaml`` reads:

    DIR/train.csv       shards 0 and 1 (128 rows)
    DIR/validation.csv  shard 2 (64 rows)

Shard 3 is the held-out final-test shard: it is verified but not written to
the data folder, so nothing in the research loop can read it.

Re-running is safe: an identical output file is kept; a different one is
refused and left unchanged. Exit status 0 on success, 2 on any refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://zenodo.org/api/records/{record}/files/{name}/content"
TRAIN_SHARDS = ("shard_0000.csv", "shard_0001.csv")
VALIDATION_SHARDS = ("shard_0002.csv",)
HEADER = "sample_id,x0,x1,x2,x3,label"


class Refused(RuntimeError):
    pass


def md5_of(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, size: int, md5: str) -> None:
    if not path.is_file():
        raise Refused(f"{path} is missing")
    if path.stat().st_size != size:
        raise Refused(f"{path}: {path.stat().st_size} bytes, data_source.json says {size}")
    if md5_of(path) != md5:
        raise Refused(f"{path}: md5 differs from data_source.json")


def download(source: dict, name: str, target: Path) -> None:
    record = source["zenodo_record_id"]
    if not record.isdigit():
        raise Refused(f"zenodo_record_id {record!r} is a placeholder; use --source_dir")
    url = API.format(record=record, name=urllib.parse.quote(name))
    # Restricted records: a Zenodo secret link carries ?token=...; it is passed on
    # to the file request. (Whether the files API honours it is not verified here.)
    if source.get("share_link"):
        token = urllib.parse.parse_qs(urllib.parse.urlparse(source["share_link"]).query).get("token")
        if token:
            url += "?" + urllib.parse.urlencode({"token": token[0]})
    print(f"downloading {name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "siderius-hackathon-fetch/1"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as sink:
        shutil.copyfileobj(response, sink)


def concatenate(paths: list[Path]) -> bytes:
    lines: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8").splitlines()
        if not text or text[0] != HEADER:
            raise Refused(f"{path.name}: header is not {HEADER!r}")
        lines.extend(text[1:])
    return ("\n".join([HEADER, *lines]) + "\n").encode("utf-8")


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() == payload:
            print(f"kept {path} (already identical)")
            return
        raise Refused(f"{path} exists with different content; left unchanged")
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Download, verify and prepare the task data.")
    cli.add_argument("--output_dir", type=Path, required=True, help="the data folder to write")
    cli.add_argument("--source_dir", type=Path, help="folder holding the Zenodo files already")
    args = cli.parse_args(argv)
    source = json.loads((HERE / "data_source.json").read_text(encoding="utf-8"))
    files = {entry["name"]: entry for entry in source["files"]}
    try:
        with tempfile.TemporaryDirectory() as scratch:
            raw = Path(args.source_dir) if args.source_dir else Path(scratch)
            for name, entry in files.items():
                if args.source_dir is None:
                    download(source, name, raw / name)
                verify(raw / name, entry["size"], entry["md5"])
            print(f"verified {len(files)} file(s) against data_source.json")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_once(args.output_dir / "train.csv", concatenate([raw / n for n in TRAIN_SHARDS]))
            write_once(args.output_dir / "validation.csv", concatenate([raw / n for n in VALIDATION_SHARDS]))
    except (Refused, OSError) as error:
        print(f"fetch_data refused: {error}", file=sys.stderr)
        return 2
    print(f"Data ready in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
