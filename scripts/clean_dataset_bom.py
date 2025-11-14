import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: clean_dataset_bom.py <path-to-json>")
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print(f"File not found: {p}")
    sys.exit(2)

b = p.read_bytes()
# Remove UTF-8 BOM if present
if b.startswith(b"\xef\xbb\xbf"):
    p.write_bytes(b[3:])
    print(f"Removed BOM from {p}")
else:
    print(f"No BOM found in {p}")
