import arxiv
import pandas as pd
import time

# --- CONFIGURATION ---
# arXiv category codes are specific.
QUERIES = {
    "Neuroscience": "cat:q-bio.NC",  # Neurons and Cognition
    "Bioinformatics": "cat:q-bio.GN",  # Genomics (often representative of Bioinfo)
    "Materials Science": "cat:cond-mat.mtrl-sci"  # Materials Science
}

TARGET_PER_CLASS = 310
OUTPUT_FILE = "arxiv_dataset.csv"

data = []

client = arxiv.Client(
    page_size=100,
    delay_seconds=3,  # Be polite
    num_retries=3
)

print(f">>> STARTING ARXIV HARVEST <<<")

for label, query in QUERIES.items():
    print(f"\n--- Fetching {label} ({query}) ---")

    # Construct the search
    search = arxiv.Search(
        query=query,
        max_results=TARGET_PER_CLASS,
        sort_by=arxiv.SortCriterion.SubmittedDate  # Get fresh papers
    )

    count = 0
    # The client handles pagination automatically
    for result in client.results(search):
        text = result.summary.replace("\n", " ")  # Clean line breaks

        # Basic filtering
        if len(text) > 50:
            data.append({
                "text": text,
                "label": label,
                "source_url": result.entry_id,
                "source_db": "arXiv"
            })
            count += 1
            if count % 50 == 0:
                print(f"   Collected {count}...")

    print(f"   Done. Got {count} abstracts.")

# Save
df = pd.DataFrame(data)

# Map labels to IDs (Consistent with Springer)
label_map = {'Neuroscience': 0, 'Bioinformatics': 1, 'Materials Science': 2}
df['label_id'] = df['label'].map(label_map)

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[SUCCESS] Saved {len(df)} rows to {OUTPUT_FILE}")