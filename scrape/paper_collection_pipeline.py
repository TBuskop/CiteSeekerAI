from search_scopus import run_scopus_search
from add_csv_to_chromadb import ingest_csv_to_chroma
from collect_relevant_abstracts import find_relevant_dois_from_abstracts
from download_papers import download_dois

# This script is a pipeline that collects relevant papers based on a given query and downloads them.

run_scopus_search()

ingest_csv_to_chroma()

relevant_doi_list = find_relevant_dois_from_abstracts()

download_dois(relevant_doi_list)