#A helper file often contains the custom functions you need to run your code
#It is like a halfway house between all code in you notebook/script, and making a full library
#A the helper file can also be called a module. They are a good idea as they help keep you notebooks 
#more easily readable. 
#It is still a good idea to write docstrings in your functions, as you will probably forget what they do and why


import openai
#import config  # Import your config.py file this contains you openai api key
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import tiktoken
import re
import difflib
from openai import OpenAI
# Set up the OpenAI API key from the .env file, this allows you to keep your key secret and not expose on github
#have the api key in like the below
#OPENAI_API_KEY = "my api key"

import time
from collections import deque


#
# chunking raw data
#



def chunk_text_with_overlap(text, chunk_size, overlap_size):
    """
    Splits a given text into chunks of specified size with overlapping sections.
    Overlap between the text chunks is necessary as it allows us to maintain the correct text structure better than if sentences are randomly cut in half.

    This function divides a text string into chunks, each containing a specified number of words. 
    Consecutive chunks overlap by a defined number of words. This method is particularly useful 
    for processing large texts in smaller segments while maintaining continuity across those segments.

    Parameters:
    text (str): The text to be chunked.
    chunk_size (int): The number of words in each chunk.
    overlap_size (int): The number of words to overlap between consecutive chunks.

    Returns:
    list of str: A list containing the text chunks with specified overlap.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        # Create chunk
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        # Move the index, considering the overlap
        i += (chunk_size - overlap_size)
    return chunks

##
## functions related to sending to the gpt api
##

class RateLimiter:
    """
    A class to implement rate limiting functionality, ensuring that the number of actions (or 'tokens') 
    does not exceed a specified maximum limit per minute.

    The rate limiter uses a token bucket algorithm, represented by deques (double-ended queues), 
    to keep track of the number of tokens and their timestamps within the past minute.

    Attributes:
        max_tokens_per_minute (int): The maximum number of tokens that are allowed in one minute.
        tokens_deque (deque): A deque to store the number of tokens generated in the past minute.
        timestamps_deque (deque): A deque to store the timestamps of when tokens were generated in the past minute.

    Methods:
        add_tokens(tokens: int): Adds a specified number of tokens to the rate limiter. If adding the tokens 
        would exceed the maximum limit, the method pauses execution until it is permissible to add the tokens.

        check_tokens(tokens: int) -> bool: Checks if adding a specified number of tokens would exceed the 
        maximum limit, without actually adding them. Returns True if adding the tokens would stay within the 
        limit, and False otherwise.

    Example:
        rate_limiter = RateLimiter(100)  # Rate limiter allowing 100 tokens per minute
        rate_limiter.add_tokens(50)      # Add 50 tokens
        if rate_limiter.check_tokens(60): 
            rate_limiter.add_tokens(60)  # Add 60 tokens if it doesn't exceed the limit
    """
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_deque = deque(maxlen=60) # Holds the tokens generated for the past minute.
        self.timestamps_deque = deque(maxlen=60) # Holds the timestamps of when tokens were generated.

    def add_tokens(self, tokens):
        current_time = time.time()

        # Removing tokens older than 1 minute
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        # If the number of tokens is more than the maximum limit,
        # pause execution until it comes back down below the threshold
        if sum(self.tokens_deque) + tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (current_time - self.timestamps_deque[0])
            time.sleep(sleep_time)

            # After sleeping, add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time + sleep_time)
        else:
            # If the number of tokens is less than the maximum limit,
            # add the tokens and timestamps to the deque
            self.tokens_deque.append(tokens)
            self.timestamps_deque.append(current_time)

    def check_tokens(self, tokens):
        # Function to check if adding new tokens would exceed limit, without actually adding them
        current_time = time.time()
        while self.timestamps_deque and current_time - self.timestamps_deque[0] > 60:
            self.timestamps_deque.popleft()
            self.tokens_deque.popleft()

        return sum(self.tokens_deque) + tokens <= self.max_tokens_per_minute
    


def get_model_response(prompt, system_message, rate_limiter, engine="gpt-3.5-turbo", alt_endpoint = None):
    """
    Sends a prompt to the GPT-3.5 API and retrieves the model's response, 
    while managing the rate of API requests using a provided rate limiter.

    This function concatenates a system message with the user's prompt, calculates 
    the total number of tokens, and uses a rate limiter to control the frequency of API requests.
    It handles various API-related errors and retries up to five times in case of failures.

    Parameters:
        prompt (str): The user's input text to be sent to the model.
        system_message (str): A system-generated message that precedes the user's prompt.
        rate_limiter (RateLimiter): An instance of the RateLimiter class to manage API request frequency.
        engine (str, optional): The model engine to use. Defaults to "gpt-3.5-turbo".
        al_endpoint (dict, optional): If a non OpenAI model is being used such as Hugginface, Groq, Anthropic, use a dict as 
        {'base_url':"<ENDPOINT_URL>" + "/v1/", 'api_key':"<API_TOKEN>"}

    Returns:
        str: The trimmed content of the model's response if successful, None otherwise.

    Raises:
        openai.error.RateLimitError: If the rate limit for the API is exceeded.
        openai.error.APIError: For general API-related errors.
        openai.error.Timeout: If a timeout occurs while waiting for the API response.

    Example:
        rate_limiter = RateLimiter(100)
        response = get_model_response("Hello, world!", "System: Starting session", rate_limiter)
        print(response)
    """
    #create the encoding object, this allows us to acurrately find the total number of tokens and ensure we don't go over the rate limit
    #There may be a better way of doing this
    #enc = tiktoken.encoding_for_model(engine)
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo") #counting is all done using bytpair
    #print(alt_endpoint)
    if alt_endpoint:
  
        client =  OpenAI(
                base_url = alt_endpoint['base_url'], 
                api_key = alt_endpoint['api_key'],  
                )   
    else:    

        client = OpenAI() #default is to instantiate using open url endpoint and api key
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    attempts = 0
    while attempts < 5:
        try:
            prompt_length = len(enc.encode(prompt))  
            tokens = len(enc.encode(system_message)) + prompt_length
            
            # Add tokens to rate limiter and sleep if necessary
            rate_limiter.add_tokens(tokens)
                
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
                max_tokens=4000, # should this be an argument?
                temperature=0.2,
                top_p=0.9,
            )
            #I am leaving the response in the raw form until I feel more confident the format is not
            #going to change or I understand why the openAI instructions do not match the example output
            #could be me, could be something else. I have no idea
            return response#['choices'][0]['message']['content'].strip()
            
        except openai.RateLimitError as e:
            print(f"RateLimitError encountered: {e}, waiting for a minute...")
            time.sleep(60)  # Wait for a minute before retrying
            continue  # Continue with the next iteration of the loop, thereby retrying the request
            
        except openai.APIError as e:
            print(f"APIError encountered: {e}, retrying in 5 seconds...")
            time.sleep(5)

        except openai.APITimeoutError as e:
            print(f"TimeoutError encountered: {e}, retrying in 10 seconds...")
            time.sleep(10)
            
        attempts += 1

    print("Failed to get model response after multiple attempts.")
    return None


class LoopMonitor:
    """
    A class to monitor and report the progress of a loop, providing
    estimations for completion based on the time taken for the last n iterations.

    Attributes:
    - total_iterations (int): Total number of iterations the loop will run.
    - n (int): The interval at which the class will report the loop's progress.
    - iterations (int): The current number of completed iterations.
    - last_n_start_time (float): The start time of the last n iterations.

    Methods:
    - update(i): To be called inside the loop to update the progress.
    - report(): Prints the current progress and estimations.
    - finish(): To be called after the loop to report the total elapsed time.
    """

    def __init__(self, total_iterations, n):
        """
        Initialize the LoopMonitor with total iterations and report interval.
        """
        self.total_iterations = total_iterations
        self.n = n
        self.iterations = 0

        # Initialize the time for the last n iterations
        # This is crucial because if the loop's tasks vary in execution time,
        # considering only the last n iterations provides a more current 
        # estimation for the remaining time.
        self.last_n_start_time = time.time()

    def update(self, i):
        """
        Update the progress of the loop. This should be called inside the loop.
        """
        self.iterations = i + 1

        if self.iterations % self.n == 0:
            self.report()
            # Reset the time for the next n iterations after reporting
            self.last_n_start_time = time.time()

    def report(self):
        """
        Print a report on the current progress and estimations.
        """
        elapsed_time_for_last_n = time.time() - self.last_n_start_time
        time_per_n = elapsed_time_for_last_n / self.n
        remaining_iterations = self.total_iterations - self.iterations
        remaining_time = time_per_n * remaining_iterations

        # Convert to minutes, round, and then convert back to seconds
        rounded_remaining_time_seconds = round(remaining_time / 60) * 60

        expected_finish_time = (datetime.now() + timedelta(seconds=rounded_remaining_time_seconds)).replace(second=0, microsecond=0)

        print(
            f"Loop number: {self.iterations} | "
            f"Time per iteration: {time_per_n:.4f} seconds | "
            f"Expected finish time: {expected_finish_time}"
        )

    def finish(self):
        """
        Print a report on the total elapsed time. This should be called after the loop.
        """
        total_elapsed_time = time.time() - self.last_n_start_time + (self.iterations / self.n) * (time.time() - self.last_n_start_time)
        print(f"Total elapsed time: {total_elapsed_time:.4f} seconds")



##
## after the chunks have been cleaned up, they will need to be joined back together
##
        

def preprocess_for_matching(text):
    """
    Prepares the text for matching by replacing non-word characters (excluding spaces) with underscores 
    and converting the text to lowercase. This process is designed to standardize the text format, 
    facilitating easier matching of overlapping sections between text chunks. It retains word positions 
    relative to the original text while ensuring newline characters and other non-word elements 
    do not interfere with the matching process.

    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text with non-word characters replaced by underscores and converted to lowercase.
    """
    #the regex uses " " not "\s" as using \s means that newline symbol \n is not replaced.
    #This can cause match errors as the use of newline may not be the same across the two chunks.
    return re.sub(r'[^\w ]', '_', text).lower()

def find_overlap(previous_chunk, current_chunk, overlap_size):
    """
    Identifies the overlapping part between the end of one text chunk and the beginning of another. 
    This function preprocesses both chunks and then uses a sequence matching approach to find the 
    longest contiguous matching subsequence. This overlap is then used to merge the chunks seamlessly.

    Parameters:
    previous_chunk (str): The preceding text chunk.
    current_chunk (str): The subsequent text chunk.
    overlap_size (int): The size of the section (in terms of words) from each chunk to be considered for finding the overlap.

    Returns:
    str: The identified overlap string.
    """
    # Preprocess chunks
    prev_chunk_processed = preprocess_for_matching(previous_chunk)
    curr_chunk_processed = preprocess_for_matching(current_chunk)

    # Extract potential overlapping parts
    prev_overlap_candidate = prev_chunk_processed.split()[-overlap_size:]
    curr_overlap_candidate = curr_chunk_processed.split()[:overlap_size]

    # Use SequenceMatcher to find the best match
    s = difflib.SequenceMatcher(None, prev_overlap_candidate, curr_overlap_candidate)
    match = s.find_longest_match(0, overlap_size, 0, overlap_size)

    # The overlap is the part in the current chunk that matches best with the end of the previous chunk
    overlap = curr_overlap_candidate[match.b: match.b + match.size]
    return ' '.join(overlap)

def merge_chunks(original_prev_chunk, original_curr_chunk, overlap):
    """
    Merges two text chunks based on a given overlap. The function preprocesses the chunks to find the starting
    index of the overlap in each and then merges them at these points. If the overlap is not found in either
    of the chunks, the function falls back to simple concatenation. This method ensures the continuity of text
    across chunks while preserving the original formatting and punctuation.

    Parameters:
    original_prev_chunk (str): The original preceding text chunk.
    original_curr_chunk (str): The original subsequent text chunk.
    overlap (str): The overlap string used to find the merging point in the text chunks.

    Returns:
    str: The merged text combining both chunks at the point of overlap.
    """
    # Preprocess both chunks for matching
    processed_prev_chunk = preprocess_for_matching(original_prev_chunk)
    processed_curr_chunk = preprocess_for_matching(original_curr_chunk)

    # Find the index where the overlap starts in each processed chunk
    start_index_prev = processed_prev_chunk.find(overlap)
    
    start_index_curr = processed_curr_chunk.find(overlap)

    if start_index_prev == -1 or start_index_curr == -1:
        # If the overlap is not found in one of the chunks, fallback to simple concatenation
        print('no match found concatenating strings')
        return original_prev_chunk + " " + original_curr_chunk

    # Merge the chunks using the overlap indices in the original text
    print('match found merging text')
    return original_prev_chunk[:start_index_prev] + original_curr_chunk[start_index_curr:]


