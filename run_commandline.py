import argparse
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from embeddings.your_main_script_name import process_and_predict

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and make predictions from FASTA file.")
    parser.add_argument("fasta_path", help="Path to the input FASTA file")
    parser.add_argument("--embedding_type", choices=['esm1b', 't5'], default='esm1b', help="Type of embedding to use")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for computations")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file path")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all_noogt", action="store_true", help="Run all NOOGT models")
    group.add_argument("--all_ogt", action="store_true", help="Run all OGT models")
    group.add_argument("--svr", action="store_true", help="Run SVR models for both NOOGT and OGT")
    
    args = parser.parse_args()

    # Set up model directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noogt_dir = os.path.join(base_dir, "noogt_regression_models")
    ogt_dir = os.path.join(base_dir, "ogt_regression_models")

    if args.all_noogt:
        noogt_model_dir = noogt_dir
        ogt_model_dir = None
    elif args.all_ogt:
        noogt_model_dir = None
        ogt_model_dir = ogt_dir
    elif args.svr:
        noogt_model_dir = os.path.join(noogt_dir, "svr")
        ogt_model_dir = os.path.join(ogt_dir, "svr")

    try:
        results = process_and_predict(args.fasta_path, 
                                      embedding_type=args.embedding_type, 
                                      use_gpu=args.use_gpu, 
                                      noogt_model_dir=noogt_model_dir, 
                                      ogt_model_dir=ogt_model_dir)
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()