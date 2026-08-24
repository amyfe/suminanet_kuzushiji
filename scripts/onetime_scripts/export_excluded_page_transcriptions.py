"""
export_excluded_page_transcriptions.py
---------------------------------------
One-time export of the ground-truth transcription for every page held out
via config.EXCLUDE_PAGES (the 20 test pages carved out by
add_excluded_books.py). These pages are never trained/validated on, so
their hand-labeled annotation JSONs (assets/data/annotations/<stem>.json)
are the closest thing to a clean reference transcription -- useful for
manually comparing against what the model actually predicts on these
held-out pages.

For each page this:
  1. Loads its annotation JSON (boxes + "U+XXXX" Unicode labels).
  2. Uses the labels in their raw annotation-array order directly, rather
     than re-deriving reading order geometrically via ROIReadingOrder --
     annotators record characters in reading order already, and empirically
     that raw order is reliable while re-sorting via ROIReadingOrder.
     sort_single was found to *corrupt* it on some pages instead (a short
     horizontal caption and at least one dense vertical page, confirmed).
     NOTE: utils/__init__.py's KuzushijiDataset loader still constructs
     actual Stage 2 training targets via this same sort_single call --
     this script no longer matches that path, and whether real training
     targets are similarly affected is an open, higher-stakes question this
     fix does not address (see THESIS_SUPERVISOR_NOTES.tex).
  3. Converts each "U+XXXX" label to its actual character and joins them.

Writes a Python-dict-literal .txt file mapping page stem -> transcription,
e.g.:
    "200021925_00003_2": "...",

Usage:
    python scripts/onetime_scripts/export_excluded_page_transcriptions.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, EXCLUDE_PAGES

ANNOT_DIR = DATA_DIR / "annotations"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "results" / "excluded_pages_transcriptions.txt"


def unicode_label_to_char(label: str) -> str:
    return chr(int(label[2:], 16))


def transcribe_page(stem: str) -> str:
    annot_path = ANNOT_DIR / f"{stem}.json"
    ann = json.loads(annot_path.read_text(encoding="utf-8"))
    labels = ann.get("labels", [])
    return "".join(unicode_label_to_char(l) for l in labels)


def main():
    stems = sorted(
        Path(f).stem
        for files in EXCLUDE_PAGES.values()
        for f in files
    )

    transcriptions = {}
    for stem in stems:
        annot_path = ANNOT_DIR / f"{stem}.json"
        if not annot_path.exists():
            print(f"  [warn] no annotation JSON for {stem}, skipping")
            continue
        transcriptions[stem] = transcribe_page(stem)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("EXCLUDED_PAGES_TRANSCRIPTIONS = {\n")
        for stem, text in transcriptions.items():
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    "{stem}": "{escaped}",\n')
        f.write("}\n")

    print(f"Wrote {len(transcriptions)} transcriptions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
