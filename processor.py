#
# FILE: processor.py
#
"""
MAYO CLINIC DATA PROCESSOR - LOCAL ONLY

This script processes raw scraped data from 'mayo_progress.json'
and outputs cleaned keywords into JSON, Excel, and CSV.

- Combines 'symptoms' + 'causes' into one field.
- Deduplicates diseases.
- Saves 'mayo_keywords_final.json', 'mayo_keywords_final.xlsx', and 'mayo_keywords_final.csv'.
"""

import os
import json
import pandas as pd
from tqdm import tqdm

INPUT_FILE = "mayo_progress.json"
MAX_KEYWORDS = 15

def merge_and_rank_keywords(*lists, top_k=MAX_KEYWORDS):
    """Merge lists of keywords while removing duplicates."""
    out, seen = [], set()
    for lst in lists:
        for it in lst or []:
            itc = str(it).strip()
            if itc and itc.lower() not in seen:
                seen.add(itc.lower())
                out.append(itc)
    return out[:top_k]

def process_scraped_data(input_data):
    print("--- Starting Local Data Processing ---")

    processed_data = []
    for url, data in tqdm(input_data.items(), desc="Processing Scraped Data"):
        if not data or not data.get("disease"):
            continue

        symptoms_causes_kws = merge_and_rank_keywords(
            data.get("symptoms", []),
            data.get("causes", [])
        )

        final_row = {
            "disease": data.get("disease"),
            "symptoms_causes": symptoms_causes_kws,
            "diagnosis": data.get("diagnosis", []),
            "tests": data.get("tests", []),
            "treatment": data.get("treatment", []),
            "prevention": data.get("prevention", []),
            "source_url": data.get("source_url")
        }
        processed_data.append(final_row)

    print("Deduplicating and sorting data...")
    unique_diseases = {}
    for item in processed_data:
        key = item["disease"].lower()
        if key not in unique_diseases:
            unique_diseases[key] = item

    final_list = sorted(unique_diseases.values(), key=lambda x: x["disease"])

    print("Formatting lists as strings...")
    for item in final_list:
        for k in ("symptoms_causes", "diagnosis", "tests", "treatment", "prevention"):
            item[k] = "; ".join(item.get(k, []))

    return final_list

def main():
    print("=== Mayo Clinic Processor - Step 2: Local Data Cleaning ===")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: '{INPUT_FILE}' not found. Run the scraper first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    final_list = process_scraped_data(input_data)
    if not final_list:
        print("No data processed.")
        return

    json_out = "mayo_keywords_final.json"
    xlsx_out = "mayo_keywords_final.xlsx"
    # --- CHANGE 1: Added CSV filename ---
    csv_out = "mayo_keywords_final.csv"

    df = pd.DataFrame(final_list)
    df = df[["disease", "symptoms_causes", "diagnosis", "tests", "treatment", "prevention", "source_url"]]
    
    # --- CHANGE 2: Added command to save the DataFrame to a CSV file ---
    df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    
    df.to_excel(xlsx_out, index=False)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(final_list)} unique diseases to:")
    print(f"- {json_out}")
    print(f"- {xlsx_out}")
    # --- CHANGE 3: Updated the print message to include the CSV file ---
    print(f"- {csv_out}")

if __name__ == "__main__":
    main()