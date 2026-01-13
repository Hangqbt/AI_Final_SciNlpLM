Hello Professor!

Below is the list of all the main files and their purpose clearly labeled for your reference!

DIRECTORIES
 |- Models
        |- ElangovanGuhan_aifinal.py -> SciBERT Main File
        |- ElangovanGuhan_aifinal_baselines.py -> Baselines Main File (TextCNN & Bi-LSTM)
        |- matrix_gen_scibert.py -> contains code for SciBERT Confusion Matrix Generation
        |- matrix_gen_baselines.py -> contains code for Confusion Matrix Generation for the baselines
        |- results_baselines_professional.xlsx -> Contains the main 5-fold test results for the baselines
        |- results_scibert_professional.xlsx -> Contains the main 5-fold test results for SciBERT
        |- arxiv_dataset.csv -> noisy, arXiv dataset that contains abstracts from pre-print copies of papers
        |- clean_dataset.csv -> springer dataset which contains abstracts from high quality peer-reviewed papers
 |- DS_Visualizations
        |- gen_gap.py -> Program to generate the generalization gap and the class distribution graphs
        |- viz_clean_ds.py -> Program to generate abstract token count distribution graphs used to determine MAX_LEN
        |- vocab_freq.py -> Program to generate a graph for the vocabulary frequency and overlap between topic labels
 |- DS_Cleaner
        |- clean_data.py -> This program cleans the dataset that was build automatically by checking for duplicates,
                            and ensure the balance of classes before use with the models