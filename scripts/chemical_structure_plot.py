#!/usr/bin/env python3

"""Draw a labeled grid of chemical structures from a SMILES list."""

import argparse
import csv
from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_DIR / "reports" / "figures" / "chemical_structures_PCP.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw chemical structures from a CSV/TSV table or a plain-text SMILES list. "
            "Plain-text lines may contain '<SMILES> <label>'."
        )
    )
    parser.add_argument("input", type=Path, help="Input CSV, TSV, .smi, or text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG or PDF path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--smiles-column",
        help="SMILES column in a CSV/TSV file (automatically detected by default).",
    )
    parser.add_argument(
        "--name-column",
        help="Optional compound-name column to use as structure labels.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        help="Structures per row (default: automatically create a near-square grid).",
    )
    parser.add_argument(
        "--cell-width", type=int, default=350, help="Cell width in pixels (default: 350)."
    )
    parser.add_argument(
        "--cell-height", type=int, default=300, help="Cell height in pixels (default: 300)."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300).")
    parser.add_argument(
        "--font-size", type=int, default=24, help="Compound-label font size (default: 24)."
    )
    return parser.parse_args()


def resolve_column(
    fieldnames: list[str], requested: str | None, candidates: tuple[str, ...]
) -> str:
    """Resolve a requested or conventional column name case-insensitively."""
    by_casefold = {name.casefold(): name for name in fieldnames}
    if requested:
        resolved = by_casefold.get(requested.casefold())
        if resolved is None:
            raise ValueError(f"Column '{requested}' was not found")
        return resolved
    for candidate in candidates:
        if candidate.casefold() in by_casefold:
            return by_casefold[candidate.casefold()]
    raise ValueError(
        "Could not detect a SMILES column; use --smiles-column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def read_table(
    path: Path, smiles_column: str | None, name_column: str | None
) -> list[tuple[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        smiles_key = resolve_column(
            reader.fieldnames, smiles_column, ("SMILES", "smiles", "canonical_smiles")
        )
        name_key = None
        if name_column:
            name_key = resolve_column(reader.fieldnames, name_column, ())
        else:
            for candidate in ("name", "compound", "compound_name", "code_name", "ID"):
                matches = [
                    name for name in reader.fieldnames if name.casefold() == candidate.casefold()
                ]
                if matches:
                    name_key = matches[0]
                    break

        records = []
        for row_number, row in enumerate(reader, start=2):
            smiles = (row.get(smiles_key) or "").strip()
            label = (row.get(name_key) or "").strip() if name_key else f"Compound {row_number - 1}"
            if smiles:
                records.append((smiles, label))
        return records


def read_text(path: Path) -> list[tuple[str, str]]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(maxsplit=1)
            label = fields[1].strip() if len(fields) == 2 else f"Compound {line_number}"
            records.append((fields[0], label))
    return records


def load_molecules(args: argparse.Namespace) -> tuple[list[Chem.Mol], list[str]]:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.suffix.lower() in {".csv", ".tsv", ".tab"}:
        records = read_table(args.input, args.smiles_column, args.name_column)
    else:
        records = read_text(args.input)

    molecules = []
    labels = []
    for index, (smiles, label) in enumerate(records, start=1):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"Warning: skipping invalid SMILES at record {index}: {smiles}", file=sys.stderr)
            continue
        rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
        molecules.append(molecule)
        labels.append(label)
    if not molecules:
        raise ValueError("The input contains no valid SMILES")
    return molecules, labels


def automatic_column_count(molecule_count: int) -> int:
    """Choose the most square grid, following 2/2/1 for five and 3/3 for six."""
    candidates = []
    for columns in range(1, molecule_count + 1):
        rows = (molecule_count + columns - 1) // columns
        empty_cells = rows * columns - molecule_count
        candidates.append((abs(rows - columns), empty_cells, columns, rows))

    best_shape = min((difference, empty) for difference, empty, _, _ in candidates)
    tied_columns = [
        columns
        for difference, empty, columns, _ in candidates
        if (difference, empty) == best_shape
    ]
    # Prefer landscape for a complete grid and portrait for an incomplete final row.
    return max(tied_columns) if best_shape[1] == 0 else min(tied_columns)


def main() -> None:
    args = parse_args()
    numeric_options = [args.cell_width, args.cell_height, args.dpi, args.font_size]
    if args.columns is not None:
        numeric_options.append(args.columns)
    if min(numeric_options) <= 0:
        raise ValueError("Grid dimensions, font size, and --dpi must be greater than zero")
    if args.output.suffix.lower() not in {".png", ".pdf"}:
        raise ValueError("--output must have a .png or .pdf extension")

    molecules, labels = load_molecules(args)
    columns = args.columns or automatic_column_count(len(molecules))
    draw_options = Draw.MolDrawOptions()
    draw_options.legendFontSize = args.font_size
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=min(columns, len(molecules)),
        subImgSize=(args.cell_width, args.cell_height),
        legends=labels,
        useSVG=False,
        drawOptions=draw_options,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".pdf":
        image = image.convert("RGB")
    image.save(args.output, dpi=(args.dpi, args.dpi))
    print(f"Saved {len(molecules)} structures to {args.output}")


if __name__ == "__main__":
    main()
