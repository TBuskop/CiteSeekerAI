import argparse
import sys
import os

# --- Add project root to sys.path ---
# Ensures that modules from 'workflows' and other 'src' subdirectories can be imported correctly.
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from workflows.obtain_store_abstracts import obtain_store_abstracts
from workflows.DeepResearch_squential import run_deep_research


def main():
    # Check if any arguments were passed (sys.argv[0] is the script name)
    if len(sys.argv) == 1:
        print("No command specified. Running default sequence: obtain_abstracts followed by deep_research.")
        print("\n--- Running obtain_store_abstracts ---")
        obtain_store_abstracts()
        print("\n--- Running run_deep_research ---")
        run_deep_research()
        print("\n--- Default sequence finished ---")
        return

    parser = argparse.ArgumentParser(description="Main CLI for academic literature LLM project.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands (workflows)", required=True)

    # Subparser for obtaining and storing abstracts
    parser_obtain_abstracts = subparsers.add_parser("obtain_abstracts", help="Run the workflow to obtain and store abstracts from Scopus and build/update ChromaDB.")
    # Add any specific arguments for obtain_store_abstracts if needed in the future
    # e.g., parser_obtain_abstracts.add_argument("--force-reindex", action="store_true", help="Force re-indexing of ChromaDB.")
    parser_obtain_abstracts.set_defaults(func=lambda args: obtain_store_abstracts())

    # Subparser for the Deep Research workflow
    parser_deep_research = subparsers.add_parser("deep_research", help="Run the Deep Research sequential pipeline.")
    # Add any specific arguments for run_deep_research if needed in the future
    # e.g., parser_deep_research.add_argument("--query", type=str, help="Override the initial research question from config.")
    parser_deep_research.set_defaults(func=lambda args: run_deep_research())
    
    args = parser.parse_args()
    
    # Call the function associated with the chosen subcommand
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
