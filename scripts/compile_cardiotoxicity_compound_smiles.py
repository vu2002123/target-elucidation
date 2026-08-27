"""Compile docked cardiotoxicity compounds, SMILES, and class into one CSV."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
OUTPUT_FILE = INTERIM_DIR / "cardiotoxicity_compounds_smiles.csv"

INPUTS = (
    (INTERIM_DIR / "debby_cp_out.csv", RAW_DIR / "debby_cp.csv", "Positive"),
    (INTERIM_DIR / "debby_cp_2_out.csv", RAW_DIR / "debby_cp_2.csv", "Positive"),
    (INTERIM_DIR / "debby_cp_n_out.csv", RAW_DIR / "debby_cp_n.csv", "Negative"),
)


def main() -> None:
    rows = []
    observed_classes = {}
    for docking_file, smiles_file, cardiotoxicity_class in INPUTS:
        docking = pd.read_csv(docking_file, usecols=["Compound"], dtype="string")
        compounds = docking["Compound"].str.strip().dropna().drop_duplicates()

        smiles = pd.read_csv(smiles_file, dtype="string")
        required = {"code_name", "SMILES"}
        missing_columns = required.difference(smiles.columns)
        if missing_columns:
            raise ValueError(f"{smiles_file} is missing columns: {sorted(missing_columns)}")
        smiles["code_name"] = smiles["code_name"].str.strip()
        smiles["SMILES"] = smiles["SMILES"].str.strip()
        smiles_lookup = (
            smiles.dropna(subset=["code_name", "SMILES"])
            .drop_duplicates("code_name")
            .set_index("code_name")["SMILES"]
        )

        missing_smiles = sorted(set(compounds).difference(smiles_lookup.index))
        if missing_smiles:
            raise ValueError(
                f"{smiles_file} has no SMILES for docked compounds: {missing_smiles}"
            )

        for compound in compounds:
            previous_class = observed_classes.get(compound)
            if previous_class is not None and previous_class != cardiotoxicity_class:
                raise ValueError(
                    f"{compound} occurs in both {previous_class} and "
                    f"{cardiotoxicity_class} inputs"
                )
            observed_classes[compound] = cardiotoxicity_class
            rows.append(
                {
                    "Compound": compound,
                    "SMILES": smiles_lookup.loc[compound],
                    "Cardiotoxicity": cardiotoxicity_class,
                    "Source docking file": docking_file.name,
                }
            )

    output = pd.DataFrame(rows).drop_duplicates("Compound", keep="first")
    class_order = pd.Categorical(
        output["Cardiotoxicity"], categories=["Positive", "Negative"], ordered=True
    )
    output = (
        output.assign(_class_order=class_order)
        .sort_values(["_class_order", "Compound"], kind="stable")
        .drop(columns="_class_order")
        .reset_index(drop=True)
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(
        f"Compiled {len(output)} compounds: "
        f"{output['Cardiotoxicity'].eq('Positive').sum()} positive and "
        f"{output['Cardiotoxicity'].eq('Negative').sum()} negative."
    )
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
