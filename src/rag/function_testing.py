from my_utils.llm_interface import initialize_clients, generate_subqueries

# change run path to file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
     sys.path.insert(0, parent_dir)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

initialize_clients()
generate_subqueries("what is climate change?", model="text-embedding-004")