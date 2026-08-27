import json, re
import requests
from time import sleep
import pandas as pd
from IPython.display import display

INTERPRO_API_BASE = "https://www.ebi.ac.uk/interpro/api/"


def get_interpro_data(url):
    output = []
    last_page = False
    while url:
        attempts = 0
        while attempts < 3:
            try:
                response = requests.get(url, headers={"Accept": "application/json"})
                if response.status_code == 408:
                    attempts += 1
                    print(f"Received 408 Timeout. Retrying {attempts}/3...")
                    sleep(61)
                    continue
                elif response.status_code == 204:
                    # no data so leave loop
                    break
                response.raise_for_status()
                data = response.json()
                if data.get("results"):
                    output.extend(data.get("results"))
                else:
                    output.append(data)
                    url = data.get("next", "")
                    sleep(1)
                    break
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                break
        else:
            print("Max retries reached for URL:", url)
        break
    return output


def get_protein_annotations(results):
    if not results:
        return pd.DataFrame(columns=["InterPro ID", "Description", "Type", "Location"])
    records = []
    for entry in results:
        interpro_id = entry.get("metadata", {}).get("accession", "N/A")
        description = entry.get("metadata", {}).get("name", "N/A")
        entry_type = entry.get("metadata", {}).get("type", "N/A")
        location_strings = []
        for protein_entry in entry.get("proteins", []):  # Iterate through protein entries
            for location in protein_entry.get(
                "entry_protein_locations", []
            ):  # Iterate through location entries
                for fragment in location.get("fragments", []):  # Iterate through fragments
                    start = fragment.get("start", "N/A")
                    end = fragment.get("end", "N/A")
                    location_strings.append(f"{start}-{end}")
        location_str = ",".join(location_strings) if location_strings else "N/A"
        records.append([interpro_id, description, entry_type, location_str])
    return pd.DataFrame(records, columns=["InterPro ID", "Description", "Type", "Location"])


uniprot_id = "P16471"
main_data_type = "entry/InterPro"
secondary_data_type = f"protein/UniProt/{uniprot_id}"
url = f"{INTERPRO_API_BASE}{main_data_type}/{secondary_data_type}/?page_size=200"
print(f"InterPro API url: {url}")

# Get the data
interpro_results = get_interpro_data(url)
interpro_df = get_protein_annotations(interpro_results)

# display the DataFrame in a table format
display(interpro_df)
