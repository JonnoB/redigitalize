


import os 
import PyPDF2
import pandas as pd



##
##
## Functions for creating the test/dev/train sets
##
##

def stratified_target_sampling(df, group_col, value_col, target_value):
    """
    Perform stratified random sampling from different groups in a DataFrame until
    the sum of the total value of the sampled groups exceeds a target value.

    Sampling stops for a group when adding another randomly selected row would
    cause the total value sum of that group's sampled rows to exceed the target value.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        group_col (str): The column name indicating the group of each observation.
        value_col (str): The column name containing the value of each observation.
        target_value (float): The target value that, once exceeded, stops further sampling from a group.

    Returns:
        pandas.DataFrame: A new DataFrame containing the balanced sampled data.
    """
    sampled_indices = []

    for group, group_df in df.groupby(group_col):
        current_sum = 0
        # Shuffle the DataFrame to ensure random ordering. Keep the original index.
        shuffled_df = group_df.sample(frac=1)
        
        for index, row in shuffled_df.iterrows():
            if current_sum + row[value_col] > target_value:
                # Stop if adding this row would exceed the target value
                break
            current_sum += row[value_col]
            # Append the index of the original DataFrame
            sampled_indices.append(index)

    # Construct the DataFrame from the chosen indices using the original DataFrame's indices
    sampled_df = df.loc[sampled_indices]

    return sampled_df


def identify_file(folder, date):
    """
    Identify the largest file in a given folder that contains a specific date string in its name.

    Args:
        folder (str): The path to the folder where the files are located.
        date (str): The date string to search for in the file names.

    Returns:
        str: The name of the largest file containing the date string, or None if no such file is found.
    """
    file_list = os.listdir(folder)
    filtered_list = [file for file in file_list if date in file]
    if filtered_list:
        # Get the full path of each file
        full_paths = [os.path.join(folder, file) for file in filtered_list]
        # Get the file sizes
        file_sizes = [os.path.getsize(path) for path in full_paths]
        # Find the index of the largest file
        largest_file_index = file_sizes.index(max(file_sizes))
        # Return the largest file
        return filtered_list[largest_file_index]
    else:
        return None
    

def find_pdf_path(image_path, folder, date):
    """
    Find the path of the largest PDF file in a given folder that contains a specific date string in its name.

    Args:
        image_path (str): The base path where the folder is located.
        folder (str): The name of the folder where the PDF files are located.
        date (str): The date string to search for in the file names.

    Returns:
        str: The full path of the largest PDF file containing the date string, or None if no such file is found.
    """
    target_folder = os.path.join(image_path, folder)
    file_name = identify_file(target_folder, date)
    if file_name:
        return os.path.join(target_folder, file_name)
    else:
        return None


def extract_pages_from_pdf(source_pdf_path, page_number, output_pdf_path, page_range=0):
    """
    Extracts specific pages from a PDF based on the target page and the specified page range.
    If the page range is 0, only the target page is extracted. If the page range is 1, the target page
    and the pages before and after it (if they exist) are extracted. The function saves the extracted
    pages to a new PDF file.

    Args:
        source_pdf_path (str): Path to the source PDF file.
        page_number (int): The target page number to extract (1-indexed).
        output_pdf_path (str): Path to save the extracted pages as a PDF.
        page_range (int, optional): The range of pages to extract around the target page. Default is 0.

    Raises:
        ValueError: If no pages could be extracted from the PDF.
    """
    # Adjust for 0-indexed page numbers
    page_number -= 1

    with open(source_pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        writer = PyPDF2.PdfWriter()

        # Determine the range of pages to extract
        start_page = max(page_number - page_range, 0)
        end_page = min(page_number + page_range, num_pages - 1)

        # Add the pages to the writer
        pages_added = False
        for page in range(start_page, end_page + 1):
            if page >= 0 and page < num_pages:
                writer.add_page(reader.pages[page])
                pages_added = True

        # If no pages were added, add the specified page (if it exists)
        if not pages_added and page_number >= 0 and page_number < num_pages:
            writer.add_page(reader.pages[page_number])

        # Write the output PDF file if at least one page was added
        if writer.pages:
            with open(output_pdf_path, 'wb') as output_file:
                writer.write(output_file)
        else:
            raise ValueError("No pages could be extracted from the PDF.")
        

def process_pdfs(temp, output_folder="output_folder", verbose=False, page_range = 0):
    """
    Processes each PDF specified in the DataFrame `temp`, extracting a specific
    page and saving it to a specified output folder.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through rows in 'pdf_path' column
    for index, row in temp.iterrows():
        pdf_path = row['pdf_path']
        page_to_extract = row['page_number']  # Assume this column exists and is correctly populated
        filename = row['file_name'].replace(".txt", ".pdf")
        destination_path = os.path.join(output_folder, filename)

        # Extract and save the specified page from the PDF
        try:
            extract_pages_from_pdf(pdf_path, page_to_extract, destination_path, page_range)
            if verbose:
                print(f"Extracted page {page_to_extract + 1} from '{pdf_path}' and saved as '{filename}' to {output_folder}")
        except Exception as e:
            if verbose:
                print(f"Error processing '{pdf_path}': {e}")
        


##
##
## Functions related to classifying the documents
##
##
            

def files_to_df_func(folder_path):

    """
    Create a pandas DataFrame from text files in a folder.

    This function reads the content of all text files in the specified folder
    and creates a DataFrame with columns for the file name, content, slug,
    periodical, page number, and issue. The slug, periodical, page number, and
    issue are extracted from the standardized file name format.

    Args:
        folder_path (str): The path to the folder containing the text files.

    Returns:
        pandas.DataFrame: A DataFrame with columns for file name, content,
            slug, periodical, page number, and issue.
    """
    # Initialize an empty list to store the data from each file
    data = []

    # Iterate over all the .txt files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the contents of the file
            with open(file_path, "r") as file:
                content = file.read()
            
            # Append the file name and content to the data list
            data.append({"file_name": file_name, "content": content})

    # Create a DataFrame from the data list
    df =  pd.DataFrame(data)

    split_df = df['file_name'].str.split("_")

    df['slug'] = split_df.apply(lambda x: x[1])
    df['periodical'] = split_df.apply(lambda x: x[3])
    df['page_number'] = split_df.apply(lambda x: x[-1]).str.replace(".txt", "").astype(int)
    df['issue'] = df['file_name'].str.extract(r'issue_(.*?)_page', expand=False)

    return df