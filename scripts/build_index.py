"""CLI helper: build the vector DB without opening the UI.

Usage:  python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingest import rebuild_index

if __name__ == "__main__":
    stats = rebuild_index()
    print(f"Done! Loaded {stats['documents']} documents into {stats['chunks']} chunks.")
