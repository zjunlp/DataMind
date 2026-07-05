#!/usr/bin/env python3
"""
Prepare DDR_Bench explore/test splits for MIMIC, 10-K, and GLOBEM.

Splits entity IDs and QA pairs into explore/test sets, then builds the
corresponding data artifacts (split SQLite databases or filtered CSV
directories) next to the original source files.

Usage:
    python prepare_splits.py mimic  --source-db PATH
    python prepare_splits.py 10k    --source-db PATH
    python prepare_splits.py globem --source-dir PATH
"""

import argparse
import csv
import json
import logging
import random
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPLORE_RATIO = 0.25
SEED = 42
CHUNK_SIZE = 5000
DDR_BENCH_DIR = Path(__file__).resolve().parents[1]


def load_entity_ids(path: Path) -> List[str]:
    """Load a JSON list of entity identifiers."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = [str(x) for x in data]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def load_qa(path: Path) -> Dict:
    """Load the QA payload (must contain a 'results' list)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"Expected 'results' list in {path}")
    return payload


def resolve_metadata_paths(args) -> Tuple[Path, Path]:
    """Use DDR_Bench/data/<scenario> metadata unless explicit files are provided."""
    default_dir = DDR_BENCH_DIR / "data" / args.scenario
    ids_file = Path(args.entity_ids).expanduser().resolve() if args.entity_ids else default_dir / "entity_ids.json"
    qa_file = Path(args.qa_file).expanduser().resolve() if args.qa_file else default_dir / "qa.json"
    return ids_file, qa_file


def build_splits(ids: Sequence[str], ratio: float, seed: int) -> Tuple[set, set]:
    """Deterministically split IDs into explore and test sets."""
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n = int(round(len(shuffled) * ratio))
    n = min(max(1, n), len(shuffled) - 1) if len(shuffled) > 1 else 1
    return set(shuffled[:n]), set(shuffled[n:])


def filter_qa(ids: Sequence[str], qa: Dict, keep: set, split: str, split_unit: str) -> Tuple[List[str], Dict]:
    """Filter entity IDs and QA results to the keep set, preserving order."""
    fids = [i for i in ids if i in keep]
    results = [r for r in qa["results"] if str(r.get("entity_id", "")) in keep]
    qa_pairs = sum(len(r.get("qa_pairs", [])) for r in results)
    meta = dict(qa.get("metadata") or {})
    meta.update(
        split=split,
        split_seed=SEED,
        split_unit=split_unit,
        explore_ratio=EXPLORE_RATIO,
        entity_count=len(results),
        qa_pair_count=qa_pairs,
    )
    return fids, {"metadata": meta, "results": results}


def split_path(base: Path, split: str) -> Path:
    """Derive a split-specific path: stem_split.suffix."""
    return base.with_name(f"{base.stem}_{split}{base.suffix}")


def write_json(path: Path, data) -> None:
    """Write JSON with parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_coverage(ids: List[str], qa: Dict):
    """Ensure entity_ids.json and qa.json cover exactly the same IDs."""
    qa_ids = {str(r.get("entity_id")) for r in qa["results"]}
    if set(ids) != qa_ids:
        raise ValueError(f"entity_ids / qa.json mismatch: "
                         f"missing QA {sorted(set(ids) - qa_ids)[:5]}, "
                         f"extra QA {sorted(qa_ids - set(ids))[:5]}")


# ---- SQLite helpers (MIMIC + 10-K) ----

def get_schema_objects(conn: sqlite3.Connection, obj_type: str):
    """Query sqlite_master for tables, indexes, or views."""
    return conn.execute(
        "SELECT name, tbl_name, sql FROM sqlite_master "
        "WHERE type=? AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name",
        (obj_type,),
    ).fetchall()


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")')]


def copy_rows(src, dst, table: str, filter_col: str, filter_vals: Sequence[str]) -> int:
    """Copy rows from src to dst, filtering by filter_col if present in the table."""
    cols = get_columns(src, table)
    qt = f'"{table}"'
    qc = ", ".join(f'"{c}"' for c in cols)
    ph = ", ".join("?" for _ in cols)

    if filter_col and filter_col in cols:
        fp = ",".join("?" for _ in filter_vals)
        cur = src.execute(f'SELECT * FROM {qt} WHERE "{filter_col}" IN ({fp})',
                          tuple(str(v) for v in filter_vals))
    else:
        cur = src.execute(f"SELECT * FROM {qt}")

    n = 0
    while True:
        rows = cur.fetchmany(CHUNK_SIZE)
        if not rows:
            break
        dst.executemany(f"INSERT INTO {qt} ({qc}) VALUES ({ph})", rows)
        n += len(rows)
    return n


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode with performance pragmas."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")
    return conn


def create_split_db(src, output: Path, schemas: Dict[str, list],
                    filter_col: str, filter_vals: Sequence[str]) -> Dict[str, int]:
    """
    Create a filtered copy of the source database.

    schemas should contain keys 'table', 'index', and optionally 'view'.
    """
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    dst = sqlite3.connect(output)
    dst.execute("PRAGMA journal_mode = OFF")
    dst.execute("PRAGMA synchronous = OFF")
    dst.execute("PRAGMA temp_store = MEMORY")
    dst.execute("PRAGMA foreign_keys = OFF")
    try:
        for _, _, sql in schemas["table"]:
            dst.execute(sql)

        counts = {}
        for name, _, _ in schemas["table"]:
            counts[name] = copy_rows(src, dst, name, filter_col, filter_vals)
            logger.info(f"  {name}: {counts[name]} rows")

        for _, _, sql in schemas.get("index", []):
            dst.execute(sql)
        for _, _, sql in schemas.get("view", []):
            dst.execute(sql)

        dst.commit()
        dst.execute("VACUUM")
        dst.execute("PRAGMA optimize")
        dst.commit()
        return counts
    finally:
        dst.close()


# ---- MIMIC ----

def run_mimic(args):
    """Split MIMIC-IV database by subject_id."""
    source_db = Path(args.source_db).expanduser().resolve()
    ids_file, qa_file = resolve_metadata_paths(args)

    ids = load_entity_ids(ids_file)
    qa = load_qa(qa_file)
    validate_coverage(ids, qa)

    explore, test = build_splits(ids, EXPLORE_RATIO, SEED)
    logger.info(f"MIMIC split: explore={len(explore)}, test={len(test)}")

    src = open_readonly(source_db)
    schemas = {t: get_schema_objects(src, t) for t in ("table", "index", "view")}

    for split, selected in [("explore", explore), ("test", test)]:
        fids, fqa = filter_qa(ids, qa, selected, split, "subject_id")
        write_json(split_path(ids_file, split), fids)
        write_json(split_path(qa_file, split), fqa)

        db_out = split_path(source_db, split)
        logger.info(f"[{split}] creating {db_out}")
        create_split_db(src, db_out, schemas, "subject_id", fids)
        logger.info(f"[{split}] done — {len(fids)} patients, {fqa['metadata']['qa_pair_count']} QA pairs")

    src.close()


# ---- 10-K ----

def run_10k(args):
    """Split 10-K database by CIK into shared explore/test databases."""
    source_db = Path(args.source_db).expanduser().resolve()
    ids_file, qa_file = resolve_metadata_paths(args)

    ids = load_entity_ids(ids_file)
    qa = load_qa(qa_file)
    validate_coverage(ids, qa)

    explore, test = build_splits(ids, EXPLORE_RATIO, SEED)
    logger.info(f"10-K split: explore={len(explore)}, test={len(test)}")

    src = open_readonly(source_db)
    schemas = {t: get_schema_objects(src, t) for t in ("table", "index")}

    for split, selected in [("explore", explore), ("test", test)]:
        fids, fqa = filter_qa(ids, qa, selected, split, "cik")
        write_json(split_path(ids_file, split), fids)
        write_json(split_path(qa_file, split), fqa)

        shared_out = split_path(source_db, split)
        logger.info(f"[{split}] creating shared DB {shared_out.name}")
        create_split_db(src, shared_out, schemas, "cik", fids)

        logger.info(f"[{split}] done — {len(fids)} companies, {fqa['metadata']['qa_pair_count']} QA pairs")

    src.close()


# ---- GLOBEM ----

def filter_csv_by_pid(src_path: Path, dst_path: Path, pids: set) -> Tuple[int, int]:
    """Filter a CSV file to rows matching the given pid set."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "r", newline="", encoding="utf-8") as fin, \
         open(dst_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        if "pid" not in header:
            raise ValueError(f"No pid column in {src_path}")
        pid_idx = header.index("pid")
        # Re-number the unnamed index column if present
        idx_col = 0 if header[0] == "" else None
        writer.writerow(header)

        written, seen = 0, set()
        for row in reader:
            if len(row) <= pid_idx or row[pid_idx] not in pids:
                continue
            if idx_col is not None:
                row[idx_col] = str(written)
            writer.writerow(row)
            written += 1
            seen.add(row[pid_idx])
    return written, len(seen)


def build_dataset_summary(source: Dict, split: str, pids: List[str], csv_stats: Dict) -> Dict:
    """Regenerate dataset_summary.json for a split directory."""
    feature_files = {k: v for k, v in csv_stats.items() if k.endswith("_allday_raw.csv")}
    total_obs = next(iter(feature_files.values()))["rows"] if feature_files else 0

    info = source.get("dataset_info", {}) if isinstance(source, dict) else {}
    ds_info = {
        "participants": len(pids),
        "total_observations": total_obs,
        "time_window": info.get("time_window", "allday"),
        "normalization": info.get("normalization", "raw (original values)"),
        "total_features": info.get("total_features"),
        "split": split,
    }
    if pids and total_obs % len(pids) == 0:
        ds_info["days_per_participant"] = total_obs // len(pids)

    return {
        "dataset_info": ds_info,
        "features_by_category": (source or {}).get("features_by_category", {}),
        "files_created": (source or {}).get("files_created", {}),
        "csv_rows": csv_stats,
    }


def run_globem(args):
    """Split GLOBEM data directory by pid."""
    source_dir = Path(args.source_dir).expanduser().resolve()
    ids_file, qa_file = resolve_metadata_paths(args)

    ids = load_entity_ids(ids_file)
    qa = load_qa(qa_file)
    validate_coverage(ids, qa)

    explore, test = build_splits(ids, EXPLORE_RATIO, SEED)
    logger.info(f"GLOBEM split: explore={len(explore)}, test={len(test)}")

    for split, selected in [("explore", explore), ("test", test)]:
        fids, fqa = filter_qa(ids, qa, selected, split, "pid")
        write_json(split_path(ids_file, split), fids)
        write_json(split_path(qa_file, split), fqa)

        # Create filtered data directory
        data_out = source_dir.parent / f"{source_dir.name}_{split}"
        if data_out.exists():
            shutil.rmtree(data_out)
        data_out.mkdir(parents=True, exist_ok=True)

        pid_set = set(fids)
        csv_stats = {}
        for f in sorted(source_dir.iterdir()):
            if f.is_dir() or f.name == "dataset_summary.json":
                continue
            if f.suffix.lower() == ".csv":
                rows, n = filter_csv_by_pid(f, data_out / f.name, pid_set)
                csv_stats[f.name] = {"rows": rows, "participants": n}
                logger.info(f"  {f.name}: {rows} rows, {n} pids")
            elif f.suffix.lower() == ".json":
                shutil.copy2(f, data_out / f.name)

        # Regenerate summary with split-accurate counts
        src_summary = {}
        if (source_dir / "dataset_summary.json").exists():
            with open(source_dir / "dataset_summary.json", "r", encoding="utf-8") as fh:
                src_summary = json.load(fh)
        write_json(data_out / "dataset_summary.json",
                   build_dataset_summary(src_summary, split, fids, csv_stats))

        logger.info(f"[{split}] done — {len(fids)} users, {fqa['metadata']['qa_pair_count']} QA pairs")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare DDR_Bench explore/test splits for MIMIC, 10-K, and GLOBEM'
    )
    sub = parser.add_subparsers(dest="scenario", required=True)

    def add_common(p):
        p.add_argument("--entity-ids", help="Path to entity_ids.json; defaults to DDR_Bench/data/<scenario>/entity_ids.json")
        p.add_argument("--qa-file", help="Path to qa.json; defaults to DDR_Bench/data/<scenario>/qa.json")

    # MIMIC
    p_mimic = sub.add_parser("mimic", help="Split MIMIC-IV by subject_id")
    p_mimic.add_argument("--source-db", required=True, help="Path to mimic_iv.db")
    add_common(p_mimic)
    p_mimic.set_defaults(func=run_mimic)

    # 10-K
    p_10k = sub.add_parser("10k", help="Split 10-K by CIK")
    p_10k.add_argument("--source-db", required=True, help="Path to 10k_financial_data.db")
    add_common(p_10k)
    p_10k.set_defaults(func=run_10k)

    # GLOBEM
    p_globem = sub.add_parser("globem", help="Split GLOBEM by pid")
    p_globem.add_argument("--source-dir", required=True, help="Path to processed GLOBEM data dir")
    add_common(p_globem)
    p_globem.set_defaults(func=run_globem)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
