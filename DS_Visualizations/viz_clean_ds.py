import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional, makes the plot look much more professional


FILE_PATH = "arxiv_dataset.csv"
OUTPUT_IMAGE = "arxiv_abstract_lengths.png"


def main():
    # we load the dataset here
    print("Loading dataset...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find '{FILE_PATH}'. Make sure it is in the same folder.")
        return

    #we calculate the word count here
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

    #we specify the plot here and style it
    plt.figure(figsize=(12, 7))


    sns.set_style("whitegrid")

    #we define the colors for the different topic labels
    colors = {
        'Neuroscience': 'blue',
        'Bioinformatics': 'orange',
        'Materials Science': 'green'
    }

    # here we plot the histograms
    for label, color in colors.items():
        subset = df[df['label'] == label]
        plt.hist(
            subset['word_count'],
            bins=30,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor='black'
        )

    # 5. Here we add labels and titles
    plt.title('Distribution of Abstract Lengths by Scientific Field (Springer)', fontsize=16)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency (Number of Papers)', fontsize=12)
    plt.legend(title='Field')
    plt.axvline(256, color='red', linestyle='dashed', linewidth=2, label='Model Cutoff (256)')

    # Here we display it and save it as an output
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nHistogram saved to '{OUTPUT_IMAGE}'")
    plt.show()

    #Here we print the dataset stats
    print("\n--- DATASET STATISTICS (Copy these to your report) ---")
    stats = df.groupby('label')['word_count'].describe()
    print(stats)

    # Here we check for how much truncation is happening if its possible on the dataset when we use our MAX_LEN value of 256
    long_papers = len(df[df['word_count'] > 256])
    print(f"\nNOTE: {long_papers} papers have more than 256 words.")
    print(f"Percentage of truncated data: {(long_papers / len(df)) * 100:.2f}%")


if __name__ == "__main__":
    main()