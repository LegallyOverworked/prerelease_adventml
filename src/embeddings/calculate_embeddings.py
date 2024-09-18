import pandas as pd
from pathlib import Path
import numpy as np
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from bio_embeddings.embed import ESM1bEmbedder
from tqdm import tqdm

def get_bio_embeddings(embedder, seqs):
    return np.array([embedder.reduce_per_protein(embedder.embed(seq)) for seq in tqdm(seqs, desc="Generating embeddings with " + embedder.name)])

def main(csv_path, use_gpu=True):
    # Load data
    df = pd.read_csv(csv_path)
    # Pre-process sequences (only keep standard 20 amino acids)
    # drop rows with empty sequences
    df = df.dropna(subset=['SEQUENCE'])
    df['SEQUENCE'] = df['SEQUENCE'].str.replace('[^ACDEFGHIKLMNPQRSTVWY]', '', regex=True)
    
    # Expanded list of embedders
    embedders = [
        ProtTransT5XLU50Embedder(device="cuda" if use_gpu else "cpu"),  # Another T5 variation
        ESM1bEmbedder(device="cuda" if use_gpu else "cpu"),  # Transformer-based
    ]
    
    for embedder in embedders:
        print(f"Processing with {embedder.name}")
        # Obtain embeddings
        embeddings = get_bio_embeddings(embedder, df['SEQUENCE'].to_list())
        # Convert embeddings to a DataFrame
        embeddings_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
        # Concatenate original data with embeddings
        concatenated_df = pd.concat([df, embeddings_df], axis=1)
        # Save to new CSV
        output_csv_path = csv_path.replace('.csv', f'_bioembeddings_{embedder.name}.csv')
        concatenated_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    main('/home/u0149897/GitLab/mlfp-dev/features/data/phoptimum_id.csv', use_gpu=False)
    # main('/home/jaldert/Desktop/prediction_data/sequence_datasets/phoptimum_brenda_id_uniprot_seq.csv', use_gpu=False)

