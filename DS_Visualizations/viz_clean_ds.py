import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, makes the plot look much more professional

# --- CONFIGURATION ---
FILE_PATH = "clean_dataset.csv"
OUTPUT_IMAGE = "abstract_lengths.png"


def main():
    # 1. Load the Data
    print("Loading dataset...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find '{FILE_PATH}'. Make sure it is in the same folder.")
        return

    # 2. Calculate Word Counts
    # We split by spaces to count words
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    # 3. Setup the Plot
    plt.figure(figsize=(12, 7))

    # Use a nice style
    sns.set_style("whitegrid")

    # Define colors for your specific fields
    colors = {
        'Neuroscience': 'blue',
        'Bioinformatics': 'orange',
        'Materials Science': 'green'
    }

    # 4. Plot Histograms
    # We loop through each label to plot them on the same graph
    for label, color in colors.items():
        subset = df[df['label'] == label]
        plt.hist(
            subset['word_count'],
            bins=30,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor='black'  # Adds a border to bars for clarity
        )

    # 5. Add Labels and Titles (Mandatory for Reports)
    plt.title('Distribution of Abstract Lengths by Scientific Field', fontsize=16)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency (Number of Papers)', fontsize=12)
    plt.legend(title='Field')
    plt.axvline(256, color='red', linestyle='dashed', linewidth=2, label='Model Cutoff (256)')

    # 6. Save and Show
    plt.savefig(OUTPUT_IMAGE, dpi=300)  # dpi=300 makes it high resolution for PDFs
    print(f"\nHistogram saved to '{OUTPUT_IMAGE}'")
    plt.show()

    # 7. Print Statistics for your Report
    print("\n--- DATASET STATISTICS (Copy these to your report) ---")
    stats = df.groupby('label')['word_count'].describe()
    print(stats)

    # Check for potential data loss
    long_papers = len(df[df['word_count'] > 256])
    print(f"\nNOTE: {long_papers} papers have more than 256 words.")
    print(f"Percentage of truncated data: {(long_papers / len(df)) * 100:.2f}%")


if __name__ == "__main__":
    main()