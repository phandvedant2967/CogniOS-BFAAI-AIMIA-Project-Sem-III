import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from app.importer import import_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Import EHR data into SQLite")
    parser.add_argument("--db-path", default=config.DB_PATH, help="Path to output SQLite database")
    parser.add_argument("--raw-dir", default=config.RAW_DIR, help="Path to raw datasets directory")
    args = parser.parse_args()

    print(f"Building database: {args.db_path} from {args.raw_dir}")
    report, warnings = import_all(db_path=args.db_path, raw_dir=args.raw_dir)
    print(f"\n{'table':<24}{'source rows':>12}{'loaded rows':>13}")
    for table, (src, loaded) in report.items():
        print(f"{table:<24}{src:>12}{loaded:>13}")
    print("\nWarnings:" if warnings else "\nNo warnings. All rows loaded, all foreign keys valid.")
    for w in warnings:
        print(f"  - {w}")
    print("\nDone.")


if __name__ == "__main__":
    main()
