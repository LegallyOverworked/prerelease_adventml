import streamlit as st
import pandas as pd
import io
from Bio import SeqIO
from io import StringIO
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from embeddings.make_predictions import process_and_predict

def predict_from_sequences(sequences, embedding_type, use_gpu, noogt_model_dir, ogt_model_dir):
    # Create a temporary FASTA file
    with open('temp.fasta', 'w') as temp_fasta:
        for i, seq in enumerate(sequences):
            temp_fasta.write(f'>sequence_{i+1}\n{seq}\n')
    
    # Process and predict
    results = process_and_predict('temp.fasta', 
                                  embedding_type=embedding_type, 
                                  use_gpu=use_gpu, 
                                  noogt_model_dir=noogt_model_dir, 
                                  ogt_model_dir=ogt_model_dir)
    
    # Remove temporary file
    os.remove('temp.fasta')
    
    return results

st.title('AdventML: Advanced Enzyme Temperature Prediction')

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ('Upload FASTA file', 'Enter single sequence', 'Enter multiple sequences')
)

# File uploader for FASTA
if input_method == 'Upload FASTA file':
    uploaded_file = st.file_uploader("Choose a FASTA file", type="fasta")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        sequences = [str(record.seq) for record in SeqIO.parse(stringio, "fasta")]
        st.success(f"Uploaded {len(sequences)} sequences")

# Text input for single sequence
elif input_method == 'Enter single sequence':
    sequence = st.text_input('Enter your protein sequence:')
    if sequence:
        sequences = [sequence]

# Text area for multiple sequences
else:
    sequences_input = st.text_area('Enter multiple sequences (one per line):')
    if sequences_input:
        sequences = sequences_input.split('\n')
        sequences = [seq.strip() for seq in sequences if seq.strip()]
        st.success(f"Entered {len(sequences)} sequences")

# Embedding type selection
embedding_type = st.selectbox('Select embedding type:', ['esm1b', 't5'])

# Use GPU checkbox
use_gpu = st.checkbox('Use GPU (if available)')

# Model selection
model_type = st.radio('Select model type:', ['All NOOGT', 'All OGT', 'SVR models'])

if st.button('Predict'):
    if 'sequences' in locals() and sequences:
        # Set up model directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        noogt_dir = os.path.join(base_dir, "noogt_regression_models")
        ogt_dir = os.path.join(base_dir, "ogt_regression_models")

        if model_type == 'All NOOGT':
            noogt_model_dir = noogt_dir
            ogt_model_dir = None
        elif model_type == 'All OGT':
            noogt_model_dir = None
            ogt_model_dir = ogt_dir
        else:  # SVR models
            noogt_model_dir = os.path.join(noogt_dir, "svr")
            ogt_model_dir = os.path.join(ogt_dir, "svr")

        # Make predictions
        with st.spinner('Processing...'):
            results = predict_from_sequences(sequences, embedding_type, use_gpu, noogt_model_dir, ogt_model_dir)

        # Display results
        st.write(results)

        # Download button
        csv = results.to_csv(index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.error("Please input sequences before predicting.")

st.sidebar.markdown("""
## About AdventML
AdventML is an advanced tool for predicting catalytic temperatures for enzymes.

### How to use:
1. Choose your input method
2. Select embedding type
3. Choose whether to use GPU
4. Select the model type
5. Click 'Predict' to get results

For more information and Updates, visit [GitHub](https://github.com/LegallyOverworked/prerelease_adventml).
""")