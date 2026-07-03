"""CLI entry point.

Examples:
  python3 -m catalog_engine.cli seller create --seller acme --name "Acme Displays"
  python3 -m catalog_engine.cli run --seller acme --file report.xlsm --pilot
  python3 -m catalog_engine.cli usage --seller acme
"""

from __future__ import annotations

import argparse
import sys
import warnings

from dotenv import load_dotenv

from .exceptions import CatalogEngineError
from .models import SellerConfig
from .store import Store

warnings.filterwarnings("ignore", message="Data Validation extension")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # ANTHROPIC_API_KEY et al. from the project .env
    ap = argparse.ArgumentParser(prog="catalog_engine")
    ap.add_argument("--data-dir", default="data", help="engine data directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("seller", help="manage sellers")
    ssub = sp.add_subparsers(dest="seller_cmd", required=True)
    sc = ssub.add_parser("create", help="create or update a seller")
    sc.add_argument("--seller", required=True)
    sc.add_argument("--name", default="")
    sc.add_argument("--marketplace", default="US", choices=["US", "UK", "EU", "CA"])
    sc.add_argument("--on-limit", default="auto_trim", choices=["auto_trim", "flag"])
    sc.add_argument("--brand-voice", default="")
    sc.add_argument("--model", default="claude-opus-4-8")
    sa = ssub.add_parser("attest", help="attest a fact term for a seller")
    sa.add_argument("--seller", required=True)
    sa.add_argument("--term", required=True,
                    help="the word/phrase being attested, e.g. 'clear'")
    sa.add_argument("--note", required=True,
                    help="the attestation, e.g. 'signage products are clear acrylic'")
    sa.add_argument("--remove", action="store_true",
                    help="withdraw the attestation for --term instead")

    rp = sub.add_parser("run", help="run the pipeline on a flat file")
    rp.add_argument("--seller", required=True)
    rp.add_argument("--file", required=True)
    rp.add_argument("--pilot", action="store_true",
                    help="one representative SKU/family per product type")
    rp.add_argument("--limit", type=int, default=None)
    rp.add_argument("--force", action="store_true",
                    help="reprocess SKUs even if source data is unchanged")
    rp.add_argument("--refresh-rules", action="store_true",
                    help="re-extract category rules, bypassing the cache")
    rp.add_argument("--workers", type=int, default=8,
                    help="concurrent generation requests (1-16, default 8)")

    up = sub.add_parser("usage", help="show per-seller LLM usage")
    up.add_argument("--seller", required=True)

    args = ap.parse_args(argv)
    store = Store(args.data_dir)
    try:
        if args.cmd == "seller" and args.seller_cmd == "create":
            store.upsert_seller(SellerConfig(
                seller_id=args.seller, display_name=args.name,
                marketplace=args.marketplace, on_limit_violation=args.on_limit,
                brand_voice=args.brand_voice, generation_model=args.model,
            ))
            print(f"seller '{args.seller}' saved "
                  f"(marketplace={args.marketplace}, on_limit={args.on_limit})")
        elif args.cmd == "seller" and args.seller_cmd == "attest":
            cfg = store.get_seller(args.seller)
            if args.remove:
                cfg.attested_terms.pop(args.term.lower(), None)
                print(f"attestation for '{args.term}' withdrawn")
            else:
                cfg.attested_terms[args.term.lower()] = args.note
                print(f"attested '{args.term}': {args.note}")
            store.upsert_seller(cfg)
            print(f"seller '{args.seller}' now has "
                  f"{len(cfg.attested_terms)} attested term(s)")
        elif args.cmd == "run":
            from .pipeline import run_pipeline
            s = run_pipeline(
                seller_id=args.seller, file_path=args.file, store=store,
                pilot=args.pilot, limit=args.limit, force=args.force,
                refresh_rules=args.refresh_rules, workers=args.workers,
            )
            print(f"\nrun {s.run_id} complete")
            print(f"  rows: {s.total_rows}  generated: {s.generated}  "
                  f"needs_review: {s.needs_review}  failed: {s.failed}  "
                  f"unchanged-skipped: {s.skipped_unchanged}")
            print(f"  seller token usage to date: {s.input_tokens} in / "
                  f"{s.output_tokens} out")
            for name, path in s.outputs.items():
                print(f"  {name}: {path}")
        elif args.cmd == "usage":
            u = store.usage_totals(args.seller)
            print(f"seller '{args.seller}': {u['calls']} LLM calls, "
                  f"{u['input_tokens']} input tokens, {u['output_tokens']} output tokens")
        return 0
    except CatalogEngineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
