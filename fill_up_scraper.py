## THIS CODE CAN BE USED INCASE OF MISSING NUMBER OF DATAPOINTS IN AN EXISTING DATASET WITHOUT SCRAPING FROM SCRATCH ##



import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

# --- CONFIGURATION ---
EXISTING_FILE = "clean_dataset.csv"
OUTPUT_FILE = "refilled_dataset_uncleaned.csv"
TARGET_PER_FIELD = 310  # Aim for slightly above 300 to be safe
BASE_URL = "https://link.springer.com"

# Headers to look like a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}


def get_soup(url):
    try:
        time.sleep(random.uniform(2, 4))  # Polite delay
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"    [!] Connection error: {e}")
    return None


def extract_abstract(article_url):
    soup = get_soup(article_url)
    if not soup: return None

    # Try different extraction methods
    abstract_div = soup.find("div", {"id": "Abs1-content"})
    if not abstract_div:
        abstract_div = soup.find("div", class_="c-article-section__content")
    if not abstract_div:  # Meta tag fallback
        meta = soup.find("meta", {"name": "description"})
        return meta["content"] if meta else None

    return abstract_div.get_text(strip=True)


# --- MAIN LOGIC ---

# 1. Load Existing Data
try:
    df = pd.read_csv(EXISTING_FILE)
    print(f"Loaded existing dataset with {len(df)} rows.")

    # Create a set of existing URLs for fast checking
    existing_urls = set(df['source_url'].tolist())

    # Count how many we currently have per field
    current_counts = df['label'].value_counts().to_dict()
    all_data = df.to_dict('records')

except FileNotFoundError:
    print("Could not find clean_dataset.csv! Starting fresh.")
    existing_urls = set()
    current_counts = {}
    all_data = []

# 2. Scrape Only What is Missing
fields = ["Neuroscience", "Bioinformatics", "Materials Science"]

for field in fields:
    count = current_counts.get(field, 0)
    needed = TARGET_PER_FIELD - count

    if needed <= 0:
        print(f"\n[OK] {field} already has {count} papers. Skipping.")
        continue

    print(f"\n--- Top-Up needed for {field}: {count} found, need {needed} more ---")

    # SMART START: Start at Page 15 since we likely have the first 14 pages (290/20 approx)
    page_num = 15

    while count < TARGET_PER_FIELD:
        search_url = f"{BASE_URL}/search/page/{page_num}?query={field}&facet-content-type=%22Article%22"
        print(f"  > Scanning Page {page_num} (Have {count}/{TARGET_PER_FIELD})...")

        soup = get_soup(search_url)
        if not soup: break

        results = soup.find_all("li", class_="app-card-open") or soup.find_all("div", class_="c-card-open")

        if not results:
            print("    [!] No results found. Moving to next page...")
            page_num += 1
            continue

        for item in results:
            if count >= TARGET_PER_FIELD: break

            link_tag = item.find("a", class_="app-card-open__link") or item.find("a", class_="c-card-open__link")
            if link_tag:
                full_link = BASE_URL + link_tag['href']

                # CHECK DUPLICATE
                if full_link in existing_urls:
                    # Skip silently to save time
                    continue

                # It is new! Fetch it.
                print(f"    [+] New Paper Found! Fetching abstract...")
                abstract_text = extract_abstract(full_link)

                if abstract_text and len(abstract_text) > 50:
                    all_data.append({
                        "text": abstract_text,
                        "label": field,
                        "source_url": full_link
                    })
                    existing_urls.add(full_link)
                    count += 1
                    print(f"        -> Saved. Total: {count}")
                else:
                    print("        -> Failed (No text).")

        page_num += 1

# 3. Final Save
final_df = pd.DataFrame(all_data)
# Re-map IDs just in case
label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
final_df['label_id'] = final_df['label'].map(label_map)

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSUCCESS! Final dataset saved to {OUTPUT_FILE} with {len(final_df)} rows.")
print(final_df['label'].value_counts())