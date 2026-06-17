# target-elucidation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Pipeline to elucidate molecular targets of active compounds

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         target_elucidation and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── scripts   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes target_elucidation a Python module
    │
    ├── conformer_generation_aqme.py             <- Create SDF files from SMILES
    │
    ├── depmap_plot.py             <- Generate DepMap plot for a gene list in a chosen cancer type
    │
    ├── docking_slurm.sh             <- Submit job to HPC
    │
    ├── domain_extraction.py             <- From a PDB file, extract a new PDB file only containing residues in a certain range (e.g. res 5-100 in a 1000 residues PDB file)
    │
    ├── eBoxSize-1.1.pl             <- Calculate the docking box size based on the input compound
    │
    ├── human_proteome_uniprot_download.py             <- Download domain data for all human verified protein (UniProt ID)
    │
    ├── intersection_all.py             <- Generate UpSet plot from docking results
    │
    ├── outfile_process.py             <- From docking output csv, split into docking output for each compound
    │
    ├── pdbe_api_compound.py             <- Download ligand data for all human verified protein that has a verified domain
    │
    ├── pdbe_process.py             <- Process domain and ligand data to find which domain to keep
    │
    ├── posebuster.py             <- Use PoseBuster to check validity of the docking result
    │
    ├── pr_curve_newds.py             <- Generate auc and pr curve for docking result
    │
    ├── prepare_docking_newds.py             <- From a list of compound, generate config file for docking (on HPC)
    │
    ├── score_collection.py             <- Get docking scores from output sdf files
    │
    ├── tsv_to_csv.py             <- Convert tsv to csv
    │
    ├── validation_list.py             <- Generate txt file containing UniProt ID of targets from BindingDB and PubChem
```

--------

