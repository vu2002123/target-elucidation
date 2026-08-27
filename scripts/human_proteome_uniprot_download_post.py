#!/usr/bin/env python3

import sys
import json
import ssl
from urllib import request, parse
from urllib.error import HTTPError
from time import sleep
from pathlib import Path

# --- Parsing Utilities ---


def parse_items(items):
    if type(items) == list:
        return ",".join(items)
    return ""


def parse_member_databases(dbs):
    if type(dbs) == dict:
        return ";".join([f"{db}:{','.join(dbs[db])}" for db in dbs.keys()])
    return ""


def parse_go_terms(gos):
    if type(gos) == list:
        return ",".join([go["identifier"] for go in gos])
    return ""


def parse_locations(locations):
    if type(locations) == list:
        return ",".join(
            [
                ",".join(
                    [
                        f"{fragment['start']}..{fragment['end']}"
                        for fragment in location["fragments"]
                    ]
                )
                for location in locations
            ]
        )
    return ""


def parse_column(value, selector):
    if value is None:
        return ""
    elif "member_databases" in selector:
        return parse_member_databases(value)
    elif "go_terms" in selector:
        return parse_go_terms(value)
    elif "children" in selector:
        return parse_items(value)
    elif "locations" in selector:
        return parse_locations(value)
    return str(value)


# --- Core API Logic ---


def fetch_batch_via_post(protein_list):
    """
    Submits a list of protein IDs via POST request.
    Follows subsequent pagination links via GET requests.
    Safely parses JSON to stdout.
    """
    context = ssl._create_unverified_context()

    # Base URL for POSTing a batch of Uniprot IDs
    url = "https://www.ebi.ac.uk/interpro/api/protein/reviewed/"

    # URL-encode the list of proteins into a POST payload body
    payload_dict = {"protein": ",".join(protein_list)}
    data_payload = parse.urlencode(payload_dict).encode("utf-8")

    next_url = url
    attempts = 0

    while next_url:
        try:
            # Send payload on the first request (POST). Send None on pagination (GET).
            req_data = data_payload if next_url == url else None

            req = request.Request(
                next_url,
                data=req_data,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )

            res = request.urlopen(req, context=context)

            if res.status == 408:
                sleep(61)
                continue
            elif res.status == 204:  # No data returned
                break

            payload = json.loads(res.read().decode())
            next_url = payload.get("next")
            attempts = 0

            # Safe extraction using .get() to prevent KeyErrors
            for item in payload.get("results", []):
                metadata = item.get("metadata", {})

                sys.stdout.write(
                    parse_column(metadata.get("accession"), "metadata.accession") + "\t"
                )
                sys.stdout.write(parse_column(metadata.get("name"), "metadata.name") + "\t")
                sys.stdout.write(
                    parse_column(metadata.get("source_database"), "metadata.source_database")
                    + "\t"
                )
                sys.stdout.write(parse_column(metadata.get("type"), "metadata.type") + "\t")
                sys.stdout.write(
                    parse_column(metadata.get("integrated"), "metadata.integrated") + "\t"
                )
                sys.stdout.write(
                    parse_column(metadata.get("member_databases"), "metadata.member_databases")
                    + "\t"
                )
                sys.stdout.write(
                    parse_column(metadata.get("go_terms"), "metadata.go_terms") + "\t"
                )

                # Handle nested protein arrays securely
                proteins = item.get("proteins", [])
                first_protein = proteins[0] if proteins else {}

                sys.stdout.write(
                    parse_column(first_protein.get("accession"), "proteins[0].accession") + "\t"
                )
                sys.stdout.write(
                    parse_column(first_protein.get("protein_length"), "proteins[0].protein_length")
                    + "\t"
                )
                sys.stdout.write(
                    parse_column(
                        first_protein.get("entry_protein_locations"),
                        "proteins[0].entry_protein_locations",
                    )
                    + "\t"
                )
                sys.stdout.write("\n")

        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            else:
                if attempts < 3:
                    attempts += 1
                    sleep(61)
                    continue
                else:
                    sys.stderr.write(f"Fatal HTTP Error at URL: {next_url}\n")
                    raise e

        # Rate-limiting compliance
        if next_url:
            sleep(1)


# --- Execution Entry Point ---

if __name__ == "__main__":
    DATA_DIR = Path.home() / "target-elucidation" / "data" / "raw"

    # Iterate through numbered chunk files 00 to 40
    for i in range(41):
        filepath = DATA_DIR / f"id_chunk_{i:02d}.txt"

        if filepath.exists():
            with open(filepath, "r") as f:
                # Read lines, ignore empty lines and whitespace
                ids = [line.strip() for line in f if line.strip()]

            if ids:
                fetch_batch_via_post(ids)

            # Additional buffer between batches to prevent 429 Too Many Requests
            sleep(2)
