#!/usr/bin/env python3

"""Compare one query SMILES against a list of PDB CCD codes."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "interim" / "smiles_similarity.csv"
CCD_COLUMN_CANDIDATES = ("ccd", "ccd_code", "code", "code_name", "id")
PDBE_COMPOUND_URL = "https://www.ebi.ac.uk/pdbe/api/v2/pdb/compound/summary/{ccd_code}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate Morgan-fingerprint Tanimoto similarity between one query "
            "SMILES and PDB chemical components whose CCD codes are supplied in "
            "a CSV, TSV, or text file. CCD SMILES are retrieved from PDBe."
        )
    )
    parser.add_argument("query_smiles", help="SMILES of the reference compound.")
    parser.add_argument("input", type=Path, help="File containing PDB CCD codes.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ranked result CSV (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument("--ccd-column", help="CCD-code column for CSV/TSV input.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (default: 2).")
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=2048,
        help="Morgan fingerprint size (default: 2048).",
    )
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0,
        help="Only retain similarities at or above this value (default: 0).",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Only write the top N matches after filtering (default: all).",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=30,
        help="PDBe request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def resolve_column(
    columns: list[str],
    requested: str | None,
    candidates: tuple[str, ...],
    column_type: str,
) -> str:
    """Resolve a column case-insensitively."""
    lookup = {column.casefold(): column for column in columns}
    if requested:
        if requested.casefold() not in lookup:
            raise ValueError(f"{column_type} column '{requested}' was not found")
        return lookup[requested.casefold()]
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not detect a {column_type} column; use --ccd-column. "
        f"Available columns: {', '.join(columns)}"
    )


def read_table(
    path: Path,
    ccd_column: str | None,
) -> list[dict[str, str | int]]:
    """Read CCD codes from a CSV or TSV file."""
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        ccd_key = resolve_column(
            reader.fieldnames,
            ccd_column,
            CCD_COLUMN_CANDIDATES,
            "CCD code",
        )
        records = []
        for row_number, row in enumerate(reader, start=2):
            ccd_code = (row.get(ccd_key) or "").strip().upper()
            if not ccd_code:
                continue
            records.append({"record": row_number, "ccd_code": ccd_code})
    return records


def read_text(path: Path) -> list[dict[str, str | int]]:
    """Read one CCD code per non-empty text line."""
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(
                {
                    "record": line_number,
                    "ccd_code": line.split(maxsplit=1)[0].upper(),
                }
            )
    return records


def fetch_ccd_smiles(ccd_code: str, timeout: float, max_attempts: int = 3) -> str:
    """Retrieve a valid SMILES for one CCD code from the PDBe API."""
    ccd_code = ccd_code.strip().upper()
    if not ccd_code:
        raise ValueError("CCD code cannot be empty")
    if timeout <= 0:
        raise ValueError("--api-timeout must be greater than zero")

    url = PDBE_COMPOUND_URL.format(ccd_code=ccd_code)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "target-elucidation-smiles-similarity/1.0",
        },
    )
    payload = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as error:
            if error.code == 404:
                raise ValueError(f"CCD code '{ccd_code}' was not found by PDBe") from error
            if error.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                raise RuntimeError(
                    f"PDBe request failed for {ccd_code}: HTTP {error.code}"
                ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == max_attempts:
                raise RuntimeError(f"PDBe request failed for {ccd_code}: {error}") from error
        time.sleep(2 ** (attempt - 1))

    matching_key = next(
        (key for key in payload if key.casefold() == ccd_code.casefold()),
        None,
    )
    summaries = payload.get(matching_key, []) if matching_key else []
    if not summaries:
        raise ValueError(f"PDBe returned no compound summary for CCD code '{ccd_code}'")
    smiles_records = summaries[0].get("smiles") or []
    candidates = [
        record.get("name", "").strip()
        for record in smiles_records
        if isinstance(record, dict) and record.get("name")
    ]
    for smiles in candidates:
        if Chem.MolFromSmiles(smiles) is not None:
            return smiles
    raise ValueError(f"PDBe returned no valid SMILES for CCD code '{ccd_code}'")


def calculate_similarities(args: argparse.Namespace) -> list[dict[str, object]]:
    """Calculate and sort Tanimoto similarities for all valid input molecules."""
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.radius < 0:
        raise ValueError("--radius cannot be negative")
    if args.fingerprint_bits < 1:
        raise ValueError("--fingerprint-bits must be greater than zero")
    if not 0 <= args.minimum_similarity <= 1:
        raise ValueError("--minimum-similarity must be between 0 and 1")
    if args.top is not None and args.top < 1:
        raise ValueError("--top must be at least 1")

    query = Chem.MolFromSmiles(args.query_smiles)
    if query is None:
        raise ValueError(f"Invalid query SMILES: {args.query_smiles}")
    query_smiles = Chem.MolToSmiles(query, canonical=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.fingerprint_bits,
    )
    query_fingerprint = generator.GetFingerprint(query)

    if args.input.suffix.lower() in {".csv", ".tsv", ".tab"}:
        records = read_table(args.input, args.ccd_column)
    else:
        records = read_text(args.input)
    if not records:
        raise ValueError(f"No CCD codes found in {args.input}")

    results = []
    invalid_count = 0
    smiles_cache = {}
    for record in records:
        ccd_code = str(record["ccd_code"])
        try:
            if ccd_code not in smiles_cache:
                smiles_cache[ccd_code] = fetch_ccd_smiles(ccd_code, args.api_timeout)
            ccd_smiles = smiles_cache[ccd_code]
        except (ValueError, RuntimeError) as error:
            invalid_count += 1
            print(f"Warning: skipping {ccd_code}: {error}", file=sys.stderr)
            continue
        molecule = Chem.MolFromSmiles(ccd_smiles)
        fingerprint = generator.GetFingerprint(molecule)
        similarity = float(DataStructs.TanimotoSimilarity(query_fingerprint, fingerprint))
        if similarity < args.minimum_similarity:
            continue
        results.append(
            {
                "query_smiles": query_smiles,
                "ccd_code": ccd_code,
                "ccd_smiles": ccd_smiles,
                "canonical_ccd_smiles": Chem.MolToSmiles(molecule, canonical=True),
                "tanimoto_similarity": similarity,
                "source_record": record["record"],
            }
        )
    if not results:
        raise ValueError("No valid molecules passed the similarity threshold")

    results.sort(
        key=lambda row: (-float(row["tanimoto_similarity"]), str(row["ccd_code"]))
    )
    if args.top is not None:
        results = results[: args.top]
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    if invalid_count:
        print(f"Skipped {invalid_count} CCD codes.", file=sys.stderr)
    return results


def write_results(path: Path, results: list[dict[str, object]]) -> None:
    """Write ranked similarity results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "query_smiles",
        "ccd_code",
        "tanimoto_similarity",
        "ccd_smiles",
        "canonical_ccd_smiles",
        "source_record",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    results = calculate_similarities(args)
    write_results(args.output, results)
    print(f"Query SMILES: {results[0]['query_smiles']}")
    print(f"Compared {len(results)} PDB chemical components.")
    print(
        f"Best match: {results[0]['ccd_code']} "
        f"(Tanimoto={results[0]['tanimoto_similarity']:.4f})"
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
