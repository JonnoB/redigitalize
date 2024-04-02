""" 
This module is poorly defined relative to the helper_file module

Generally it is supposed to be functions that cause the text to be manipulated with a specific goal by the LLM's

"""

from helper_file import *

# Define a function to process each row
def process_row(row, system_message_template, rate_limiter, engine="gpt-3.5-turbo", alt_endpoint = None):
    # Extract necessary information from the row
    title = row['title']
    date = row['issue_date']
    content = row['content_html']
    
    # Format the system message with the actual title and date
    system_message = system_message_template.format(title=title, date=date)

    # Assuming get_model_response is a function that sends the content and system message to an AI model
    response = get_model_response(content, system_message, rate_limiter, engine=engine,  alt_endpoint = alt_endpoint).choices[0].message.content
    return response


def perform_capoc(df, corrected_folder , system_message_template, engine, data_path='./data',  alt_endpoint = None):

    """
    Performs Context Aware Post-Ocr Correcetion on a dataframe of text. 
    The dataframe must contain a 'title' column, indicating the name of the periodical, an 'issue_date' column with the
    the issue was printed, a content_html column with the raw OCR data.

    This function iterates through `df`, checking if each row's ID has been processed before. For new IDs, it 
    recovers the text using the `process_row` function, which applies a model-based correction to poor-quality OCR 
    content. Corrected texts are saved in individual text files within a Folder called 

    Parameters:
    - df (pandas.DataFrame): DataFrame containing OCR data to be processed. Required columns are 'title', 
      'issue_date', 'content_html', 'id', and others relevant for file naming.
    - corrected_folder (str): Descriptive name for the data subset, influencing subfolder naming.
    - engine (str, optional): The processing engine used for text recovery. Defaults to "gpt-3.5-turbo".
    - folder_path (str, optional): Project base data folder for recovered text, Defaults to './data'

    Process:
    - Checks for existing processed IDs to avoid reprocessing.
    - Recovers text from poor-quality OCR using `process_row`.
    - Saves both original and recovered text in a structured file system.

    Notes:
    - Assumes `process_row` for text recovery, which requires a row, a RateLimiter instance, and an engine name.
    - A `RateLimiter` class is presumed to manage the rate of API calls or processing tasks.
    """
    # Create new subfolder path
    new_subfolder = os.path.join(data_path, f'{corrected_folder}_{engine}')
    if not os.path.exists(new_subfolder):
        os.makedirs(new_subfolder)

    # Path for the times_df CSV file
    times_csv_path = os.path.join(new_subfolder, f"0_processing_time.csv")

    # Check if times_df CSV file exists to continue from last save
    if os.path.exists(times_csv_path):
        times_df = pd.read_csv(times_csv_path)
    else:
        times_df = pd.DataFrame(columns=['id', 'time'])

    # Convert the 'id' column to a set for faster lookup
    processed_ids = set(times_df['id'])

    # List to accumulate new records
    new_records = []

    # Instantiate rate limiter
    rate_limiter = RateLimiter(50000)
    
    for index, row in df.iterrows():
        if row['id'] not in processed_ids:
            start_time = time.time()  # Begin measuring time

            corrected_ocr = process_row(row, system_message_template, rate_limiter, engine=engine,  alt_endpoint = alt_endpoint)  # Assuming process_row is defined

            end_time = time.time()  # Stop measuring time
            elapsed_time = round(end_time - start_time)  # Time to the nearest second

            # Construct file name and path
            file_path = os.path.join(new_subfolder, row['file_name'])
            
            # Save corrected_ocr as a text file
            with open(file_path, 'w') as file:
                file.write(corrected_ocr)
            
            # Append new record to the list
            new_records.append({'id': row['id'], 'time': elapsed_time})
            
            times_df = pd.concat([times_df, pd.DataFrame(new_records)], ignore_index=True)
            times_df.to_csv(times_csv_path, index=False)

            
            # Add the processed ID to the set
            processed_ids.add(row['id'])

# Note: This code assumes that 'df' has columns 'id', 'slug_periodical', 'slug_issue', 'page_number',
# and that 'process_row' is a function defined elsewhere that takes a row, a RateLimiter instance, and an engine name as arguments.
            

def run_capoc_tasks(dev_data, configurations):
    """
    Runs Context Aware Post-OCR Correction (CAPOC) tasks on a given dataset with various configurations.

    Parameters:
    - dev_data (pandas.DataFrame): The DataFrame containing OCR data to be processed.
    - configurations (list of dicts): A list where each dict contains the parameters for a
      `perform_capoc` call, including 'corrected_folder', 'system_message', 'engine',
      and optionally 'data_path'.

    Each dict in the configurations list should have the following keys:
    - 'corrected_folder' (str): The folder name where corrected texts are saved.
    - 'system_message' (str): The system message template passed to `process_row`.
    - 'engine' (str): The processing engine used for text recovery.
    - 'data_path' (str, optional): The base data folder for saving recovered texts. Defaults to './data'.
    """
    for config in configurations:
        # Extract each configuration parameter, with a default for 'data_path'
        corrected_folder = config['corrected_folder']
        system_message = config['system_message']
        engine = config['engine']
        data_path = config.get('data_path', './data')  # Provide a default value if not specified
        alt_endpoint = config.get('alt_endpoint', None)

        # Call perform_capoc with the current configuration
        perform_capoc(dev_data, corrected_folder, system_message, engine, data_path,  alt_endpoint)


def classify_genre_row(row, rate_limiter, engine="gpt-3.5-turbo-0125",  alt_endpoint = None):
    
    enc = tiktoken.encoding_for_model(engine)
    title = row['title']
    date = row['issue_date']
    content = row['gpt4_response']
    # Truncate content to max first 500 tokens
    encoding = enc.encode(content)
    # Truncate using slicing
    max_tokens = 500  # Replace with your desired number of tokens
    truncated_encoding = encoding[:max_tokens]

    # Decode the truncated encoding back to text
    truncated_text = enc.decode(truncated_encoding)
    
    system_message = """ You are a machine that classifies newspaper articles. Your response  is limited to choices from the following json
                            {0: 'news report',
                            1: 'editorial',
                            2: 'letter',
                            3: 'advert',
                            4: 'review',
                            5: 'poem/song/story',
                            6: 'other'}
                            you will respond using a single digit.

                        For example given the text "Mr Bronson died today, he was a kind man" your answer would be
                        6
                        
                        Alternatively given the text "The prime minster spoke at parliament today" your answer would be
                        0
                        """
        
    prompt = f"""
                    Classify the following article
                    :::
                    {truncated_text}
                    :::
                    
                    """
    

    response = get_model_response(prompt, system_message, rate_limiter, engine=engine,  alt_endpoint = None).choices[0].message.content

    return response


def classify_topics_row(row, rate_limiter, engine="gpt-3.5-turbo-0125",  alt_endpoint = None):
    
    enc = tiktoken.encoding_for_model(engine)
    content = row['gpt4_response']
    # Truncate content to max first 500 tokens
    encoding = enc.encode(content)
    # Truncate using slicing
    max_tokens = 500  # Replace with your desired number of tokens
    truncated_encoding = encoding[:max_tokens]

    # Decode the truncated encoding back to text
    truncated_text = enc.decode(truncated_encoding)
    
    system_message = """ You are a machine that classifies newspaper articles. Your response  is limited to choices from the following json
                        {0: 'arts, culture, entertainment and media',
                        1: 'crime, law and justice',
                        2: 'disaster, accident and emergency incident',
                        3: 'economy, business and finance',
                        4: 'education',
                        5: 'environment',
                        6: 'health',
                        7: 'human interest',
                        8: 'labour',
                        9: 'lifestyle and leisure',
                        10: 'politics',
                        11: 'religion',
                        12: 'science and technology',
                        13: 'society',
                        14: 'sport',
                        15: 'conflict, war and peace',
                        16: 'weather'}
                        you will respond using a numeric python list.

                    For example given the text "The War with spain has forced schools to close" your answer would be
                    [15,4]
                    
                    Alternatively given the text "The prime minster spoke at parliament today" your answer would be
                    [10]
                    """
    
    prompt = f"""
                    Classify the following article
                    :::
                    {truncated_text}
                    :::
                    
                    """
    

    response = get_model_response(prompt, system_message, rate_limiter, engine=engine,  alt_endpoint = None).choices[0].message.content

    return response