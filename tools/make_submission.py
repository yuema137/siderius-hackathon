"""Build <task_id>.zip from a Track 1 submission folder.

Usage::

    python tools/make_submission.py <task_dir> [--output_dir DIR]

Runs checker step 1 (layout, sizes, file types, TASK_CARD.md, JSON schemas) on
the folder first and refuses to build if it fails. Cache and editor clutter
(__pycache__, *.pyc, .DS_Store, .git, ...) is left out. The zip is
deterministic: the same folder always gives the same bytes and sha256. An
existing output file is never replaced. Exit status 0 on success, 1 on refusal.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submission_rules import MAX_ZIP_BYTES, build_zip_bytes, check_static, sha256_bytes  # noqa: E402


def make(task_dir: Path, output_dir: Path) -> tuple[Path, int, str]:
    """Write ``output_dir/<task_id>.zip``; returns ``(path, bytes, sha256)``.

    Raises:
        ValueError: step 1 fails, the zip exceeds 20 MB, or the output exists.
    """
    root = task_dir.resolve()
    static = check_static(root)
    if static.problems:
        raise ValueError("step 1 refuses this folder:\n  - " + "\n  - ".join(static.problems))
    payload = build_zip_bytes(root, static.files)
    if len(payload) > MAX_ZIP_BYTES:
        raise ValueError(f"the zip would be {len(payload)} bytes; the limit is {MAX_ZIP_BYTES} (20 MB)")
    target = output_dir.resolve() / f"{root.name}.zip"
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise ValueError(f"{target} already exists; move it away or choose another --output_dir") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return target, len(payload), sha256_bytes(payload)


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Build <task_id>.zip from a submission folder.")
    cli.add_argument("task_dir", type=Path, help="the <task_id>/ submission folder")
    cli.add_argument("--output_dir", type=Path, default=Path.cwd(), help="existing folder (default: current)")
    args = cli.parse_args(argv)
    if not args.task_dir.is_dir():
        print(f"make_submission refused: {args.task_dir} is not a folder", file=sys.stderr)
        return 1
    if not args.output_dir.is_dir():
        print(f"make_submission refused: --output_dir {args.output_dir} is not an existing folder", file=sys.stderr)
        return 1
    try:
        target, size, digest = make(args.task_dir, args.output_dir)
    except ValueError as error:
        print(f"make_submission refused: {error}", file=sys.stderr)
        return 1
    print(f"wrote {target}")
    print(f"size {size} bytes")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
