#
# FILE: mayo_scraper.py
#
"""
MAYO CLINIC SCRAPER (V16 - FINAL QUALITY UPGRADE)

This script scrapes Mayo Clinic disease pages and extracts keywords.

✅ FINAL QUALITY UPGRADES:
- ADVANCED DEDUPLICATION: Uses lemmatization to treat plural and singular forms
  of a keyword as the same item (e.g., "symptom" vs "symptoms"), eliminating repetition.
- ENHANCED RELEVANCE: NLP model now prioritizes nouns and proper nouns, ensuring
  keywords are meaningful "things" or "concepts" and filters out irrelevant words.

✅ PREVIOUS FEATURES RETAINED:
- Correct A-Z index navigation.
- Polite scraping and stable ProcessPoolExecutor.
- High-precision extraction from bold tags and bullets.
"""

import os
import re
import json
import time
import string
from urllib.parse import urljoin
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import yake
import spacy
from tenacity import retry, wait_exponential, stop_after_attempt

# ---------------- CONFIG FOR POLITE SCRAPING ----------------
BASE_URL = "https://www.mayoclinic.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
CHECKPOINT_FILE = "mayo_progress.json"
MAX_WORKERS = 2
MAX_KEYWORDS = 20
YAKE_LANG = "en"
YAKE_MAX_NGRAM = 3

# ---------------- NLP (Initialized once per process) ----------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy 'en_core_web_sm' model not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

kw_extractor = yake.KeywordExtractor(lan=YAKE_LANG, n=YAKE_MAX_NGRAM, top=MAX_KEYWORDS)

# ---------------- UTILITY FUNCTIONS ----------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def get_url(url, timeout=20):
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    time.sleep(0.5)
    return resp

def get_disease_urls_from_index():
    print("Gathering disease URLs from A-Z index...")
    disease_urls = set()
    alphabet = string.ascii_uppercase
    for letter in tqdm(alphabet, desc="Scanning A-Z Index"):
        index_url = f"https://www.mayoclinic.org/diseases-conditions/index?letter={letter}"
        try:
            r = get_url(index_url)
            soup = BeautifulSoup(r.text, 'html.parser')
            result_divs = soup.find_all('div', class_='cmp-result-name')
            if result_divs:
                for div in result_divs:
                    link = div.find('a', href=True)
                    if link and '/diseases-conditions/' in link['href']:
                        disease_urls.add(urljoin(BASE_URL, link['href']))
        except Exception as e:
            print(f"Could not process index page for letter {letter}: {e}")
    return sorted(list(disease_urls))


def extract_keywords_from_prose(text, top_k=5):
    """UPGRADED: NLP extraction with relevance filtering (nouns/proper nouns)."""
    if not text or len(text) < 3:
        return []
        
    doc = nlp(text)
    
    # Filter for meaningful nouns and proper nouns
    meaningful_chunks = []
    for chunk in doc.noun_chunks:
        # A noun chunk is meaningful if its root word is a noun or proper noun
        if chunk.root.pos_ in ['NOUN', 'PROPN']:
            # Exclude chunks that are just pronouns or numbers
            if not chunk.root.is_stop and not chunk.root.like_num and not chunk.root.is_punct:
                 meaningful_chunks.append(chunk.text.strip())

    # Use YAKE for additional keyword candidates
    temp_kw_extractor = yake.KeywordExtractor(lan=YAKE_LANG, n=YAKE_MAX_NGRAM, top=top_k, dedupLim=0.9)
    yake_kws = [k for k, _ in temp_kw_extractor.extract_keywords(text)]

    # Combine and deduplicate
    out, seen_lemmas = [], set()
    for kw_list in [meaningful_chunks, yake_kws]:
        for kw in kw_list:
            # Lemmatize to handle plurals and variations for deduplication
            kw_doc = nlp(kw.lower())
            lemma = " ".join([token.lemma_ for token in kw_doc])
            
            if len(kw) > 2 and lemma not in seen_lemmas:
                seen_lemmas.add(lemma)
                out.append(kw)

    return out[:top_k]


def scrape_sub_page(url):
    page_keywords = {}
    try:
        r = get_url(url)
        soup = BeautifulSoup(r.text, "html.parser")
        for heading in soup.find_all(re.compile("^h[2-4]$")):
            section_title = heading.get_text(separator=" ").strip().lower()
            if not section_title:
                continue

            section_content_elements = []
            for sib in heading.find_next_siblings():
                if sib.name and re.match("^h[2-4]$", sib.name):
                    break
                if hasattr(sib, 'name') and sib.name:
                    section_content_elements.append(sib)

            keywords = {} # Use dict to store keyword and its lemma for deduplication
            paragraph_text = []

            for element in section_content_elements:
                # 1. Direct extraction from bold tags
                for bold_tag in element.find_all(['b', 'strong']):
                    text = bold_tag.get_text(" ", strip=True)
                    if text and len(text.split()) <= 5:
                        lemma = " ".join([token.lemma_ for token in nlp(text.lower())])
                        if lemma not in keywords:
                            keywords[lemma] = text

                # 2. NLP extraction on each individual list item
                for li_tag in element.find_all('li'):
                    text = li_tag.get_text(" ", strip=True)
                    if text:
                        bullet_keywords = extract_keywords_from_prose(text, top_k=3)
                        for kw in bullet_keywords:
                            lemma = " ".join([token.lemma_ for token in nlp(kw.lower())])
                            if lemma not in keywords:
                                keywords[lemma] = kw

                # 3. Collect plain paragraph text
                if element.name == 'p':
                    paragraph_text.append(element.get_text(" ", strip=True))

            # 4. Run NLP on paragraph text as a fallback
            if len(keywords) < 7 and paragraph_text:
                prose_text = " ".join(paragraph_text)
                nlp_keywords = extract_keywords_from_prose(prose_text, top_k=10)
                for kw in nlp_keywords:
                    lemma = " ".join([token.lemma_ for token in nlp(kw.lower())])
                    if lemma not in keywords:
                        keywords[lemma] = kw

            final_list = sorted(list(keywords.values()), key=str.lower)

            if final_list:
                page_keywords[section_title] = final_list[:MAX_KEYWORDS]

    except Exception:
        pass
    return page_keywords


def scrape_disease_worker(base_url):
    try:
        r = get_url(base_url)
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("h1")
        disease_name = " ".join(title_tag.get_text().split()) if title_tag else base_url.split("/")[-1].replace("-", " ").title()
        row_data = {"disease": disease_name, "source_url": base_url}
        links = soup.find_all("a", href=True)
        symptoms_url = next((urljoin(BASE_URL, a["href"]) for a in links if "symptoms-causes" in a["href"]), None)
        diagnosis_url = next((urljoin(BASE_URL, a["href"]) for a in links if "diagnosis-treatment" in a["href"]), None)
        if not symptoms_url and not diagnosis_url:
            return base_url, None, f"Skipped (no sub-links): {disease_name}"
        all_keywords = {}
        if symptoms_url:
            all_keywords.update(scrape_sub_page(symptoms_url))
        if diagnosis_url:
            all_keywords.update(scrape_sub_page(diagnosis_url))
            
        def get_section(kws, *keywords):
            combined_kws = {} # Use dict for lemmatized deduplication
            for key, value in kws.items():
                if any(kw in key for kw in keywords):
                    for v in value:
                        lemma = " ".join([token.lemma_ for token in nlp(v.lower())])
                        if lemma not in combined_kws:
                            combined_kws[lemma] = v
            return sorted(list(combined_kws.values()), key=str.lower)

        row_data["symptoms"] = get_section(all_keywords, "symptom", "sign")
        row_data["causes"] = get_section(all_keywords, "cause", "risk factor", "trigger")
        row_data["diagnosis"] = get_section(all_keywords, "diagnos", "exam")
        row_data["treatment"] = get_section(all_keywords, "treatment", "therap", "manage")
        row_data["tests"] = get_section(all_keywords, "test", "screen")
        row_data["prevention"] = get_section(all_keywords, "prevention", "lifestyle", "reduc")
        if any(row_data.get(k) for k in ["symptoms", "causes", "diagnosis", "treatment", "tests", "prevention"]):
            return base_url, row_data, "Success"
        else:
            return base_url, None, f"Skipped (no keywords extracted): {disease_name}"
    except Exception as e:
        return base_url, None, f"FAILED | Error: {type(e).__name__}"

# ---------------- MAIN EXECUTION ----------------
def main():
    print("=== Mayo Clinic Scraper - Step 1: Data Collection ===")
    all_urls = get_disease_urls_from_index()
    print(f"Found {len(all_urls)} disease URLs from the A-Z index.")
    if not all_urls:
        print("No URLs found. The HTML structure of the index pages may have changed. Exiting.")
        return
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        checkpoint_data = {}
    urls_to_scrape = [u for u in all_urls if u not in checkpoint_data]
    if not urls_to_scrape:
        print("All URLs already scraped. Nothing new to do.")
        return
    print(f"Starting to scrape {len(urls_to_scrape)} new URLs...")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(scrape_disease_worker, url): url for url in urls_to_scrape}
        with tqdm(total=len(urls_to_scrape), desc="Scraping Disease Pages") as pbar:
            for future in as_completed(future_to_url):
                url, row_data, status = future.result()
                if row_data:
                    checkpoint_data[url] = row_data
                    if len(checkpoint_data) % 10 == 0:
                        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                            json.dump(checkpoint_data, f, indent=2)
                if "Skipped" in status or "FAILED" in status:
                    pbar.set_postfix_str(status, refresh=True)
                pbar.update(1)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"\n✅ Scraping complete. Total diseases scraped: {len(checkpoint_data)}")
    print(f"Raw data saved to '{CHECKPOINT_FILE}'.")

if __name__ == "__main__":
    main()