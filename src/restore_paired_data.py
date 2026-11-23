"""Extract the complete SPY paired dataset from the saved Colab .ipynb file."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def restore(notebook, destination):
    book = json.loads(Path(notebook).read_text(encoding="utf-8"))
    blocks = {}
    for cell in book["cells"]:
        for output in cell.get("outputs", []):
            record = output.get("data", {}).get("application/json")
            if isinstance(record, dict) and record.get("format") == "SPY-pair-data-v1":
                blocks[int(record["part"])] = record
    if not blocks or set(blocks) != set(range(blocks[0]["count"])):
        raise ValueError("The notebook must be the saved copy with all dataset blocks.")
    digest = blocks[0]["sha256"]
    if any(row["sha256"] != digest for row in blocks.values()):
        raise ValueError("Mismatched dataset versions")
    raw = b"".join(base64.b64decode(blocks[i]["payload"], validate=True) for i in range(len(blocks)))
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Dataset checksum mismatch")
    Path(destination).write_bytes(raw)
    print(f"Saved {len(raw):,} verified bytes to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("output", nargs="?", default="SPY_Paired_2020_2025_Data.zip")
    args = parser.parse_args()
    restore(args.notebook, args.output)
