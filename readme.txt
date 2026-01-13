Hello Professor!

Below is the list of all the main files and their purpose clearly labeled for your reference!

NOTE: Other misc. Source codes which were used in data aquisition are available in my GitHub Repository along with the main codes included here!

Repository Link: https://github.com/Hangqbt/AI_Final_SciNlpLM 

Python Version: 3.11

######################################################################################################################################################################

DIRECTORIES
 |- Models
 |       |- ElangovanGuhan_aifinal.py -> SciBERT Main File
 |       |- ElangovanGuhan_aifinal_baselines.py -> Baselines Main File (TextCNN & Bi-LSTM)
 |       |- matrix_gen_scibert.py -> contains code for SciBERT Confusion Matrix Generation
 |       |- matrix_gen_baselines.py -> contains code for Confusion Matrix Generation for the baselines
 |       |- results_baselines_professional.xlsx -> Contains the main 5-fold test results for the baselines
 |       |- results_scibert_professional.xlsx -> Contains the main 5-fold test results for SciBERT
 |       |- arxiv_dataset.csv -> noisy, arXiv dataset that contains abstracts from pre-print copies of papers
 |       |- clean_dataset.csv -> springer dataset which contains abstracts from high quality peer-reviewed papers
 |		 |
 |		 |- Ablation_Testing
 |		 |	|- ablation_main_1_0.py -> This contains the code to automatically run the ablation tests in the SciBERT Model with different configurations
 |		 |	|- hyperparameter_tuning_results.xlsx -> This contains the results of the ablation testing of the model
 |		 |	
 |		 |- Conf_Matrix_Output
 |			 | - confusion_matrix_Bi-LSTM_arxiv.png -> Confusion Matrix Image of the Bi-LSTM Model
 |			 | - confusion_matrix_TextCNN_arxiv.png -> Confusion Matrix Image of the TextCNN Model
 |			 | - confusion_matrix_scibert.png -> Confusion Matrix Image of the SciBERT Model
 |			
 |- DS_Visualizations
 |       |- gen_gap.py -> Program to generate the generalization gap and the class distribution graphs
 |       |- viz_clean_ds.py -> Program to generate abstract token count distribution graphs used to determine MAX_LEN
 |       |- vocab_freq.py -> Program to generate a graph for the vocabulary frequency and overlap between topic labels
 |- DS_Cleaner
         |- clean_data.py -> This program cleans the dataset that was build automatically by checking for duplicates,
                             and ensure the balance of classes before use with the models
							
							
######################################################################################################################################################################