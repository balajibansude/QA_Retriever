import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.loader import load_all_pdfs

docs = load_all_pdfs("data")
