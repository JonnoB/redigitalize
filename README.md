# Improving Accessibility to 19th-Century Newspapers with Large Language Models

This repo is part of a project to help increase access to the 19th Century digital archive the 19th Century serials edition.

# Process

There are two main parts to the project, shown below

## OCR correction
The project takes and SQL database of OCR data for a variety of publications.
- Converts the SQL database into a set of 7 parquet files with a subset of metadata columns
- Creates 3 subsets
    - a dev set 
    - a test set of 100 articles and 125 adverts
    - a silver train set of a mixture 1000 adverts and articles
- Checks 8 different prompt models plus 2 prompts 
- Using the the top performing prompt, compares 7 models using two prompt and OCR text combinations
- Takes the top performing model and prompt arrangement, and performs post-OCR correction on the entire dataset 

## Article labelling

The below process is used to label the articles into two different class types; 
- Genre, which describes if the text is 
    - article 
    - advert 
    - editorial
    - etc. 
- Subjec: Using the IPTC NewsCodes schema

The process is as follows

- Takes the corrected articles
- Compares LLM performance for the same prompt at classifying articles into classes
- Classifies the dataset using the most sucessful model

# Data

The main dataset for the project is the 19th Century Serials edition website Postgres dump file available from the King's College Figshare at xxxx

# Using this repo

The code in this repo has been broken up into several small(ish) ipynb each one working on a different aspect of data manipulation and project process.

The approximate order of the notebooks is as follows

- explore_sql: Looks at the NCSE database and extracts it into a series of parquet files
- ecplore_data: Basic look at the distributions in the data
- create_test_set: Create the dev, test, and silver label train sets
- prompt_testiing: Exploring different priompts for correcting the OCR
- classification: The process for creatintg the training set as well as training the classification models
- analysis: The notebook used to create the results tables and figures used in the paper

There are also some python scripts which act as modules containing the majority of the functions used in the project

- llm_comparison_toolkit: contains all the functions used to set the up the calls to the LLM's testing there cleaning/classification abilities. This is supposed to be relatively generic and re-usable
- evaluation_funcs: contains the functions used in the evaluation of the output of the LLM's



# Citing this project