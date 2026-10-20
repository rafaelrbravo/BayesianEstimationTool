"""Copy bundled BET examples and Markdown/PDF documentation into an editable folder."""
import argparse
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Copy bundled BET examples and documentation.")
    parser.add_argument("path", nargs="?", default=".", help="Parent directory for BET-Examples (default: current directory)")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    source = package / "Examples"
    docs = package / "docs"
    parent = Path(args.path).expanduser().resolve()
    target = parent / "BET-Examples"
    if not parent.is_dir():
        print(f"Parent directory does not exist: {parent}", file=sys.stderr)
        return 1
    if not source.is_dir() or not list(source.glob("*.py")):
        print("BET examples are missing from this installation.", file=sys.stderr)
        return 1
    documents = sorted(p for p in docs.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".md"}) if docs.is_dir() else []
    if not documents:
        print("BET documentation Markdown/PDF files are missing from this installation.", file=sys.stderr)
        return 1
    if target.exists():
        print(f"Destination already exists: {target}\\nNothing was overwritten.", file=sys.stderr)
        return 1
    shutil.copytree(source, target)
    docs_target = target / "Documentation"
    docs_target.mkdir()
    for document in documents:
        shutil.copy2(document, docs_target / document.name)
    print(f"Copied BET examples and documentation to {target}")
    print("Run an example with: python BET-Examples/1_SimpleExampleNuts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
