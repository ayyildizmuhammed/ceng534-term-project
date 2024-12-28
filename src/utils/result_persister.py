# src/utils/result_persister.py

import csv
import json
import os

def save_results_to_csv(data, filename="results.csv"):
    """
    Saves a list of dictionaries to a CSV file.
    
    :param data: A list of dicts, where each dict contains the fields to be saved.
    :param filename: Name of the output CSV file (default: results.csv).
    """
    if not data:
        print("No data to save.")
        return

    # Get all unique keys from the data
    keys = set().union(*(d.keys() for d in data))

    with open(filename, mode="w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def save_results_to_json(data, filename="results.json"):
    """
    Saves a list of dictionaries to a JSON file.
    
    :param data: A list of dicts
    :param filename: Name of the output JSON file
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
