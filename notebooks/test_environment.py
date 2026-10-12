import sys

print("Python:")
print(sys.executable)

print("\nTesting packages....")

import pandas
print("Pandas:", pandas.__version__)

import sentence_transformers
print("sentence-transformers: OK")

import chromadb
print("chromadb: OK")

