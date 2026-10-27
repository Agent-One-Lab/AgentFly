"""Build the pyserini JsonCollection (documents.jsonl) for the WebShop Lucene index.

Uses the SAME load_products the server uses, so doc ids (asins) and the searchable
`contents` text exactly match what the running server expects. Dataset is chosen via
the WEBSHOP_DATASET env var ("full" or "small"), matching the server. Run at image
build time; the resulting index is baked in and never rebuilt at runtime.
"""
import os
import json

from utils import init_basedir, get_file_path
from engine import load_products

dataset = os.environ.get("WEBSHOP_DATASET", "full")
init_basedir(dataset=dataset)

print(f"[build_index] dataset={dataset}: loading products ...", flush=True)
all_products, *_ = load_products(filepath=get_file_path())
print(f"[build_index] loaded {len(all_products)} products; writing documents.jsonl", flush=True)

os.makedirs("resources", exist_ok=True)
n = 0
with open("resources/documents.jsonl", "w") as f:
    for p in all_products:
        opts = []
        for name, contents in (p.get("options") or {}).items():
            try:
                opts.append(f"{name}: {', '.join(contents)}")
            except Exception:
                pass
        option_text = ", and ".join(opts)
        bp = p.get("BulletPoints") or [""]
        bullet = bp[0] if bp else ""
        contents = " ".join([
            str(p.get("Title") or ""),
            str(p.get("Description") or ""),
            str(bullet or ""),
            option_text,
        ]).lower()
        f.write(json.dumps({"id": p["asin"], "contents": contents}) + "\n")
        n += 1

print(f"[build_index] wrote {n} docs to resources/documents.jsonl", flush=True)
