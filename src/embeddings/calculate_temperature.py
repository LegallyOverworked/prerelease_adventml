import pandas as pd
import numpy as np
from Bio import SeqIO
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from bio_embeddings.embed import ESM1bEmbedder
from tqdm import tqdm

def get_bio_embeddings(embedder, seqs):
    return np.array([embedder.reduce_per_protein(embedder.embed(seq)) for seq in tqdm(seqs, desc="Generating embeddings with " + embedder.name)])

def process_fasta_embeddings(fasta_path, embedding_type='esm1b', use_gpu=True):
    # Load sequences from FASTA file
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    
    # Pre-process sequences (only keep standard 20 amino acids)
    sequences = [''.join(filter(lambda x: x in 'ACDEFGHIKLMNPQRSTVWY', seq)) for seq in sequences]
    
    # Select embedder
    if embedding_type.lower() == 'esm1b':
        embedder = ESM1bEmbedder(device="cuda" if use_gpu else "cpu")
    elif embedding_type.lower() in ['t5', 'prottranst5xlu50']:
        embedder = ProtTransT5XLU50Embedder(device="cuda" if use_gpu else "cpu")
    else:
        raise ValueError("Invalid embedding type. Choose 'esm1b' or 't5'.")
    
    # Obtain embeddings
    embeddings = get_bio_embeddings(embedder, sequences)
    
    # Create DataFrame
    df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df.insert(0, 'id', ids)  # Add ID column at the beginning
    
    return df

# Example usage:
# df = process_fasta_embeddings('path/to/your/fasta/file.fasta', embedding_type='esm1b', use_gpu=False)
# df.to_csv('output_embeddings.csv', index=False)