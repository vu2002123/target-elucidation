#!/usr/bin/env bash

pdb_id="$1"
lig_code="$2"
chain_id="$3"

pdb_id="${pdb_id^^}"
lig_code="${lig_code^^}"
chain_id="${chain_id^^}"

pdb_file="${pdb_id}.pdb"
cif_file="${pdb_id}.cif"
cognate_file="${pdb_id}_${chain_id}_${lig_code}_cognate.pdb"
rec_raw_file="${pdb_id}_${chain_id}_rec_raw.pdb"
rec_file="${pdb_id}_${chain_id}_rec.pdb"
rec_h_file="${pdb_id}_${chain_id}_rec_H.pdb"

require_nonempty() {
    local output_file="$1"
    local description="$2"

    if [[ ! -s "$output_file" ]]; then
        echo "Error: ${description} is empty or was not generated: ${output_file}" >&2
        return 1
    fi
}

download_cif() {
    if [[ -s "$cif_file" ]]; then
        return 0
    fi
    wget -O "$cif_file" "https://files.rcsb.org/download/${pdb_id}.cif" && \
        [[ -s "$cif_file" ]]
}

if ! wget -O "$pdb_file" \
    "https://files.rcsb.org/download/${pdb_id}.pdb" || [[ ! -s "$pdb_file" ]]; then
    echo "PDB download failed for ${pdb_id}; trying the CIF structure." >&2
    rm -f "$pdb_file"

    if ! download_cif; then
        echo "Error: could not download a PDB or CIF structure for ${pdb_id}." >&2
        exit 1
    fi

    if ! obabel "$cif_file" -O "$pdb_file" || [[ ! -s "$pdb_file" ]]; then
        echo "Error: could not convert ${cif_file} to PDB with obabel." >&2
        exit 1
    fi
fi

# Extract ATOM records from the selected protein chain.
awk -v chain="$chain_id" '
    substr($0,1,6) == "ATOM  " &&
    substr($0,22,1) == chain
' "$pdb_file" > "${pdb_id}_${chain_id}_rec_raw.pdb"
if ! require_nonempty "$rec_raw_file" "raw receptor file"; then
    echo "Check that chain ${chain_id} exists and contains ATOM records in ${pdb_id}." >&2
    exit 1
fi

# Write to a different file to avoid overwriting the input.
if ! obabel "$rec_raw_file" -O "$rec_file" ||
    ! require_nonempty "$rec_file" "converted receptor file"; then
    exit 1
fi

# Extract the specified cognate ligand from the selected chain.
# awk -v lig="$lig_code" -v chain="$chain_id" '
#     substr($0,1,6) == "HETATM" &&
#     substr($0,18,3) == lig &&
#     substr($0,22,1) == chain
# ' "${pdb_id}.pdb" > "${pdb_id}_${chain_id}_${lig_code}_cognate.pdb"

# PDB residue names are limited to three characters. Use mmCIF for longer CCD
# identifiers, and also whenever the original PDB download required a CIF fallback.
if [[ ${#lig_code} -gt 3 || -s "$cif_file" ]]; then
    ligand_cif="${pdb_id}_${chain_id}_${lig_code}_cognate.cif"
    if ! download_cif; then
        echo "Error: CIF is required to extract CCD ${lig_code}, but download failed." >&2
        exit 1
    fi

    if ! python - "$cif_file" "$ligand_cif" "$chain_id" "$lig_code" <<'PY'
import sys

import gemmi


source_file, output_file, requested_chain, requested_ligand = sys.argv[1:]
structure = gemmi.read_structure(source_file)
selected = gemmi.Structure()
selected.name = f"{structure.name}_{requested_ligand}"
selected.cell = structure.cell
selected.spacegroup_hm = structure.spacegroup_hm
output_model = gemmi.Model("1")
output_chain = gemmi.Chain(requested_chain)
match_count = 0

for model in structure:
    for chain in model:
        if chain.name.upper() != requested_chain.upper():
            continue
        for residue in chain:
            if residue.name.upper() != requested_ligand.upper():
                continue
            output_residue = gemmi.Residue()
            output_residue.name = residue.name
            output_residue.seqid = residue.seqid
            output_residue.subchain = residue.subchain
            output_residue.entity_type = residue.entity_type
            for atom in residue:
                if atom.altloc in ("\x00", " ", "A"):
                    output_residue.add_atom(atom.clone())
            if len(output_residue):
                output_chain.add_residue(output_residue)
                match_count += 1

if match_count == 0:
    raise SystemExit(
        f"CCD {requested_ligand} was not found in chain {requested_chain} of {source_file}"
    )

output_model.add_chain(output_chain)
selected.add_model(output_model)
selected.make_mmcif_document().write_file(output_file)
PY
    then
        echo "Error: could not extract CCD ${lig_code} from chain ${chain_id}." >&2
        exit 1
    fi

    if ! obabel "$ligand_cif" -O "$cognate_file" || [[ ! -s "$cognate_file" ]]; then
        echo "Error: could not convert extracted ligand ${ligand_cif} to PDB." >&2
        exit 1
    fi
    rm -f "$ligand_cif"
else
    awk -v lig="$lig_code" -v chain="$chain_id" -v alt="A" '
        substr($0,1,6) == "HETATM" &&
        substr($0,18,3) == lig &&
        substr($0,22,1) == chain &&
        (substr($0,17,1) == alt || substr($0,17,1) == " ") {
            print substr($0,1,16) " " substr($0,18)
        }
    ' "$pdb_file" > "$cognate_file"
fi
if ! require_nonempty "$cognate_file" "cognate ligand file"; then
    echo "Check that ligand ${lig_code} exists in chain ${chain_id} of ${pdb_id}." >&2
    exit 1
fi

# Add hydrogens to the receptor.
if ! reduce -BUILD "$rec_file" > "$rec_h_file" ||
    ! require_nonempty "$rec_h_file" "hydrogenated receptor file"; then
    exit 1
fi

rm "$rec_raw_file"
rm -f "$cif_file"
