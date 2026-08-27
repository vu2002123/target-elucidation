#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


def protonate_ligand(folder, compound):
    ligand_in = os.path.join(folder, f"{compound}_start_conf.sdf")
    ligand_out = os.path.join(folder, f"{compound}_start_conf_H.sdf")
    subprocess.run(["obabel", ligand_in, "-O", ligand_out, "-p", "7.4"])


def generate_conformers(folder, compound, num_confs=10):
    mol = Chem.MolFromMolFile(os.path.join(folder, f"{compound}_start_conf_H.sdf"))
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, os.path.join(folder, "single", f"{compound}.sdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        help="dir containing compound sdf file",
    )
    parser.add_argument(
        "--compound_name",
        type=str,
        help="name of compound",
    )
    args = parser.parse_args()

    protonate_ligand(args.base_path, args.compound_name)
    generate_conformers(args.base_path, args.compound_name)
