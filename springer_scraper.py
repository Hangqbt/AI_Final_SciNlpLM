import requests
from bs4 import BeautifulSoup
import time
import random
import pandas as pd

# --- CONFIGURATION ---
TARGET_FIELDS = ["Neuroscience", "Bioinformatics", "Materials Science"]
TARGET_COUNT = 300  # Abstracts per field
BASE_URL = "https://link.springer.com"
OUTPUT_FILE = "ai_assignment_dataset.csv"

# Mimic a real browser to avoid immediate anti-bot blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}


def get_soup(url):
    """
    Helper function to get BeautifulSoup object from a URL.
    Includes error handling and polite delays.
    """
    try:
        # RANDOM DELAY: Crucial to look like a human reading papers
        sleep_sec = random.uniform(3, 6)
        time.sleep(sleep_sec)

        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser")
        else:
            print(f"    [!] Error {response.status_code} fetching {url}")
            return None
    except Exception as e:
        print(f"    [!] Connection failed: {e}")
        return None


def extract_abstract_from_page(article_url):
    """
    Visits the individual article page and extracts the full abstract
    using your specific logic + fallbacks.
    """
    soup = get_soup(article_url)
    if not soup:
        return None

    # STRATEGY 1: Standard Springer ID (Your code)
    abstract_div = soup.find("div", {"id": "Abs1-content"})

    # STRATEGY 2: Class-based (Common in newer layouts)
    if not abstract_div:
        abstract_div = soup.find("div", class_="c-article-section__content")

    # STRATEGY 3: Meta Description (The safety net)
    if not abstract_div:
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            return meta_desc["content"]

    if abstract_div:
        return abstract_div.get_text(strip=True)

    return None


def scrape_field(field_name):
    """
    Main logic to scrape N abstracts for a specific field.
    """
    print(f"\n--- Starting Collection for: {field_name} ---")
    collected_data = []
    page_num = 1

    while len(collected_data) < TARGET_COUNT:
        # Springer Search URL Structure: /search/page/X?query=KEYWORD
        search_url = f"{BASE_URL}/search/page/{page_num}?query={field_name}&facet-content-type=%22Article%22"
        print(f"  > Scanning Search Page {page_num} ({len(collected_data)}/{TARGET_COUNT} collected)...")

        soup = get_soup(search_url)
        if not soup:
            break

        # Find all article links on the search results page
        # Springer typically uses 'a' tags with class 'app-card-open__link' or 'title' inside results
        results = soup.find_all("li", class_="app-card-open")

        if not results:
            # Fallback for different Springer layout variants
            results = soup.find_all("div", class_="c-card-open")

        if not results:
            print("    [!] No results found on this page. Crawler might be blocked or end of results.")
            break

        for item in results:
            if len(collected_data) >= TARGET_COUNT:
                break

            # Find the link to the full article
            link_tag = item.find("a", class_="app-card-open__link") or item.find("a", class_="c-card-open__link")

            if link_tag and 'href' in link_tag.attrs:
                full_link = BASE_URL + link_tag['href']

                # Visit the article page to get the abstract
                abstract_text = extract_abstract_from_page(full_link)

                if abstract_text and len(abstract_text) > 50:  # Filter out empty/bad abstracts
                    collected_data.append({
                        "text": abstract_text,
                        "label": field_name,
                        "source_url": full_link  # Good to keep for reference
                    })
                    print(f"    [+] Collected abstract ({len(abstract_text)} chars)")
                else:
                    print(f"    [-] Skipped (No abstract found)")

        page_num += 1

    return collected_data


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    all_abstracts = []

    for field in TARGET_FIELDS:
        field_data = scrape_field(field)
        all_abstracts.extend(field_data)

        # Save progress after each field (Safety mechanism)
        temp_df = pd.DataFrame(all_abstracts)
        temp_df.to_csv(f"partial_{OUTPUT_FILE}", index=False)
        print(f"--- Finished {field}. Saved partial progress. ---\n")

    # Final Save
    final_df = pd.DataFrame(all_abstracts)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDONE! Successfully collected {len(final_df)} abstracts.")
    print(f"Data saved to {OUTPUT_FILE}")