from bioservices import InterPro, UniProt

i = InterPro()
u = UniProt()

uniprot_id = "P00533"
results = i.get_protein_entries(uniprot_id)
results = u.mapping
superfamilies = [
    {
        "accession": entry["metadata"]["accession"],
        "name": entry["metadata"]["name"],
        "source_database": entry["metadata"]["source_database"],
    }
    for entry in results.get("results", [])
    if entry.get("metadata", {}).get("type") == "homologous_superfamily"
]
# 3. Display results
for sf in superfamilies:
    print(f"Superfamily: {sf['name']} ({sf['accession']}) from {sf['source_database']}")


entry = i.get_entry("IPR000001")
entry["metadata"]["name"]

print("Accession:", entry["metadata"]["accession"])
print("Name     :", entry["metadata"]["name"])
print("Type     :", entry["metadata"]["type"])

results = i.get_entries(page_size=5, page=1)
print("Total entries:", results["count"])
for r in results["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

results = i.search_entries("kinase", page_size=5)
print("Matches for 'kinase':", results["count"])
for r in results["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

results = i.get_entries_by_type("domain", page_size=5)
print("Total domain entries:", results["count"])
for r in results["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

pfam_entry = i.get_member_database_entry("pfam", "PF00001")
print("Pfam entry name:", pfam_entry["metadata"]["name"])

pfam_results = i.get_entries_by_member_database("pfam", page_size=5)
print("Total Pfam entries:", pfam_results["count"])
for r in pfam_results["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

protein = i.get_protein("P00734")
print("Protein name:", protein["metadata"]["name"])
print("Source      :", protein["metadata"]["source_database"])
print("Length      :", protein["metadata"]["length"])

entries = i.get_protein_entries("P00734")
print("InterPro entries for P00734:")
for r in entries["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

proteins = i.get_proteins_by_entry("IPR000001", page_size=5)
print("Total proteins for IPR000001:", proteins["count"])
for r in proteins["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

structure = i.get_structure("1t2v")
print("PDB ID  :", structure["metadata"]["accession"])
print("Name    :", structure["metadata"]["name"])

structures = i.get_entry_structures("IPR000001", page_size=5)
print("Total structures for IPR000001:", structures["count"])
for r in structures["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

taxon = i.get_taxonomy("9606")
print("Scientific name:", taxon["metadata"]["name"]["name"])
print("Full name      :", taxon["metadata"]["name"]["short"])

taxons = i.get_entry_taxonomy("IPR000001", page_size=5)
print("Total taxa for IPR000001:", taxons["count"])
for r in taxons["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

proteome = i.get_proteome("UP000005640")
print("Proteome accession:", proteome["metadata"]["accession"])
print("Name              :", proteome["metadata"]["name"])
print("Is reference      :", proteome["metadata"]["is_reference"])

proteomes = i.get_entry_proteomes("IPR000001", page_size=5)
print("Total proteomes for IPR000001:", proteomes["count"])
for r in proteomes["results"]:
    print(r["metadata"]["accession"], "-", r["metadata"]["name"])

pfam_clan = i.get_set("pfam", "CL0001")
print("Clan accession:", pfam_clan["metadata"]["accession"])
print("Clan name     :", pfam_clan["metadata"]["name"])
