import pandas as pd
import numpy as np
from Bio import SeqIO
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from bio_embeddings.embed import ESM1bEmbedder
from tqdm import tqdm
import joblib
import os

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

def load_models(model_dir):
    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith('.joblib') and 'train' not in filename:
            model_path = os.path.join(model_dir, filename)
            model_name = os.path.splitext(filename)[0]
            models[model_name] = joblib.load(model_path)
    return models

def make_predictions(df, models):
    X = df.filter(regex='^emb_')  # Select only embedding columns
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X)
    return pd.DataFrame(predictions)

def process_and_predict(fasta_path, embedding_type='esm1b', use_gpu=True, noogt_model_dir=None, ogt_model_dir=None):
    # Generate embeddings
    df = process_fasta_embeddings(fasta_path, embedding_type, use_gpu)
    
    results = df[['id']].copy()
    
    # Make predictions for NOOGT models
    if noogt_model_dir:
        noogt_models = load_models(noogt_model_dir)
        noogt_predictions = make_predictions(df, noogt_models)
        results = pd.concat([results, noogt_predictions], axis=1)
    
    # Make predictions for OGT models
    if ogt_model_dir:
        ogt_models = load_models(ogt_model_dir)
        if 'ogt' not in df.columns:
            raise ValueError("OGT column is required for OGT predictions but not found in the data.")
        X_ogt = df.filter(regex='^emb_')
        X_ogt['ogt'] = df['ogt']
        ogt_predictions = make_predictions(X_ogt, ogt_models)
        results = pd.concat([results, ogt_predictions], axis=1)
    
    return results

# Example usage:
# results = process_and_predict('path/to/your/fasta/file.fasta', 
#                               embedding_type='esm1b', 
#                               use_gpu=False, 
#                               noogt_model_dir='path/to/noogt_regression_models', 
#                               ogt_model_dir='path/to/ogt_regression_models')
# results.to_csv('output_predictions.csv', index=False)