"""Restore a SPY v2 checkpoint ZIP from a saved Colab backup notebook.

Run: python restore_checkpoints.py SPY-EMA-V2_All_8_Checkpoints.ipynb
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib
from zipfile import ZipFile


ARCHIVE = "SPY-EMA-V2-checkpoints.zip"


def restore(notebook, destination):
    saved = json.loads(notebook.read_text(encoding="utf-8"))
    chunks = {}
    total = None
    expected_digest = None
    for cell in saved["cells"]:
        for output in cell.get("outputs", []):
            for line in "".join(output.get("text", [])).splitlines():
                if line.startswith("Checkpoint backup: ") and "; SHA-256 " in line:
                    expected_digest = line.split("; SHA-256 ", 1)[1].split(";", 1)[0]
            data = output.get("data", {}).get("application/json")
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, dict) or data.get("archive") != ARCHIVE:
                continue
            if data.get("encoding") != "zlib+base64":
                raise ValueError("Unrecognized checkpoint encoding")
            index, count = data["part"], data["parts"]
            if total not in (None, count) or not 1 <= index <= count or index in chunks:
                raise ValueError("Inconsistent or duplicate archive part")
            chunks[index] = base64.b64decode(data["data"], validate=True)
            total = count
    if not chunks or len(chunks) != total or not expected_digest:
        raise ValueError("Incomplete archive. Select the full checkpoint backup notebook.")
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        decoder = zlib.decompressobj()
        with temp.open("wb") as out:
            for i in range(1, total + 1):
                block = decoder.decompress(chunks[i])
                out.write(block)
                digest.update(block)
            tail = decoder.flush()
            out.write(tail)
            digest.update(tail)
        if not decoder.eof or decoder.unused_data or digest.hexdigest() != expected_digest:
            raise ValueError("Archive SHA-256 verification failed")
        with ZipFile(temp) as archive:
            if archive.testzip() is not None:
                raise ValueError("Checkpoint ZIP integrity check failed")
        if destination.exists():
            with destination.open("rb") as existing:
                present = hashlib.sha256()
                for block in iter(lambda: existing.read(1 << 20), b""):
                    present.update(block)
            if present.hexdigest() != expected_digest:
                raise FileExistsError(f"Different file already exists at {destination}")
        else:
            # Exclusive creation keeps a concurrent restore from overwriting another file.
            os.link(temp, destination)
        print(f"{destination}  SHA-256 {expected_digest}")
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, default=Path(ARCHIVE))
    args = parser.parse_args()
    restore(args.notebook, args.output)
