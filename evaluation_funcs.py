import jellyfish
import numpy as np
import pandas as pd
import os

def vectorized_levenshtein_similarity(leven_distances, lengths):
    """
    Vectorized calculation of Levenshtein similarity based on arrays of Levenshtein distances
    and the corresponding lengths of strings.

    Parameters:
    - leven_distances (np.array): An array of Levenshtein distances.
    - lengths (np.array): An array of the maximum lengths between compared strings.

    Returns:
    - np.array: An array of Levenshtein similarities.
    """
    # Ensure zero division is handled by returning 0 similarity in such cases
    similarities = np.where(lengths > 0, (lengths - leven_distances) / lengths, 0)
    return similarities

# Example usage within the calculate_metrics function
def calculate_metrics(file_name, raw_data_string, contents, wer, cer):
    wer_score = wer.compute(predictions=[raw_data_string], references=[contents])
    cer_score = cer.compute(predictions=[raw_data_string], references=[contents])
    leven_score = jellyfish.levenshtein_distance(raw_data_string, contents)
    
    # Here we prepare for vectorized computation
    leven_distances = np.array([leven_score])
    lengths = np.array([max(len(raw_data_string), len(contents))])
    
    lev_sim = vectorized_levenshtein_similarity(leven_distances, lengths)[0]  # Get the first element since we're dealing with single values

    results_df = pd.DataFrame({
        'File Name': [file_name],
        'WER': [wer_score],
        'CER': [cer_score],
        'lev_dist': [leven_score],
        'lev_sim': [lev_sim]
    })

    return results_df


def process_dataset(dev_data_raw_df, dev_transcripts, wer, cer):
    """
    Processes a dataset to calculate Word Error Rate (WER), Character Error Rate (CER),
    and Levenshtein Distance for each entry. It assumes each entry includes a file name
    pointing to a transcript file, and raw data content for comparison.

    WARNING: Performs no pre-processing like convert to lower, remove certain symbols etc

    Parameters:
    - dev_data_raw_df (pd.DataFrame): A DataFrame containing the dataset to be processed.
      It must include 'file_name' and 'content_html' columns.
    - dev_transcripts (str): The directory path where the transcript files referenced in
      dev_data_raw_df are stored.

    Returns:
    - pd.DataFrame: A combined DataFrame containing the file name, WER, CER, and Levenshtein
      Distance for each file in the dataset.

    This function iterates through each row in the input DataFrame, reads the transcript file
    contents, and uses the `calculate_metrics` function to calculate the necessary metrics. It
    compiles the results from all files into a single DataFrame.
    """

    results_list = []  # To store the dataframes generated for each file
    
    # Loop through each row in the DataFrame
    for index, row in dev_data_raw_df.iterrows():
        file_name = row['file_name']
        raw_data_string = row['content_html']
        file_path = os.path.join(dev_transcripts, file_name)

        # Open the file and read its contents
        with open(file_path, 'r', encoding='utf-8') as file:
            contents = file.read()

        # Calculate metrics and append the resulting dataframe to the list
        metrics_df = calculate_metrics(file_name, raw_data_string.lower(), contents.lower(), wer, cer)
        results_list.append(metrics_df)
    
    # Combine all the DataFrames in the list into a single DataFrame
    combined_df = pd.concat(results_list, ignore_index=True)
    
    return combined_df


def load_txt_files_to_df(directory):
    """
    Loads the content of all '.txt' files within a specified directory into a pandas DataFrame.

    Parameters:
    - directory (str): The path to the directory containing '.txt' files.

    Returns:
    - pd.DataFrame: A DataFrame with a single column "content_html", where each row contains
      the content of one '.txt' file from the directory.
    """
    # Initialize a list to store the content of each text file
    content_list = []
    
    # Loop through each file in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a '.txt' file
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory, file_name)
            # Open the file and read its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content_list.append({'content_html': content, 'file_name':file_name})
    
    # Create a DataFrame with the contents
    df = pd.DataFrame(content_list)
    
    return df