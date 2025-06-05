#!/bin/bash
# Step-by-step tutorial for using the PLM framework with Protein Gym data
# This script demonstrates the complete workflow from data preparation to model evaluation

set -e  # Exit on error

# Create directories
mkdir -p tutorial_results/s22a1/{data,models,plots,proposals}

# Step 1: Define the reference sequence (S22A1 protein)
echo "Step 1: Setting up reference sequence"
S22A1_SEQUENCE="PTVDDILEQVGESGWFQKQAFLILCLLSAAFAPICVGIVFLGFTPDHHCQSPGVAELSQRCGWSPAEELNYTVPGLGPAGEAFLGQCRRYEVDWNQSALSCVDPLASLATNRSHLPLGPCQDGWVYDTPGSSIVTEFNLVCADSWKLDLFQSCLNAGFLFGSLGVGYFADRFGRKLCLLGTVLVNAVSGVLMAFSPNYMSMLLFRLLQGLVSKGNWMAGYTLITEFVGSGSRRTVAIMYQMAFTVGLVALTGLAYALPHWRWLQLAVSLPTFLFLLYYWCVPESPRWLLSQKRNTEAIKIMDHIAQKNGKLPPADLKMLSLEEDVTEKLSPSFADLFRTPRLRKRTFILMYLWFTDSVLYQGLILHMGATSGNLYLDFLYSALVEIPGAFIALITIDRVGRIYPMAMSNLLAGAACLVMIFISPDLHWLNIIIMCVGRMGITIAIQMICLVNAELYPTFVRNLGVMVCSSLCDIGGIITPFIVFRLREVWQALPLILFAVLGLLAAGVTLLLPETKGVALPETMKDAENLGRKAKPKENTIYLKVQTSEPSGT"

# Save reference sequence to a file
echo ">S22A1_reference" > tutorial_results/s22a1/data/reference.fasta
echo "$S22A1_SEQUENCE" >> tutorial_results/s22a1/data/reference.fasta
echo "Reference sequence saved to tutorial_results/s22a1/data/reference.fasta"

# Step 2: Prepare the dataset
echo -e "\nStep 2: Preparing the dataset"
# Copy the Protein Gym dataset to our working directory
cp test_data/S22A1_HUMAN_Yee_2023_activity.csv tutorial_results/s22a1/data/
echo "Dataset copied to tutorial_results/s22a1/data/"

# Step 3: Initialize the PLM framework
echo -e "\nStep 3: Initializing the PLM framework"
# Create a custom configuration file for this tutorial
cat > tutorial_results/s22a1/config.yaml << EOF
model:
  name1: facebook/esm2_t33_650M_UR50D
  name: facebook/esm2_t6_8M_UR50D
  embedding_dim1: 1280
  embedding_dim: 320
  reduced_dim: null
  use_pooling: true
  quantize: false
training:
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 1.0e-05
  epochs: 50
  early_stopping: 10
  device: auto
active_learning:
  acquisition: ucb
  batch_size: 10
  temperature: 1.0
  exploration_weight: 2.0
  diversity_weight: 0.5
data:
  db_path: tutorial_results/s22a1/data/variants.db
  embedding_cache: tutorial_results/s22a1/data/embeddings.h5
  output_dir: tutorial_results/s22a1/results
EOF
echo "Configuration file created at tutorial_results/s22a1/config.yaml"

# Step 4: Embedding the reference sequence
echo -e "\nStep 4: Embedding the reference sequence"
python -c "
from plm_framework.cli import embed_sequence
try:
    embed_sequence(
        sequence='$S22A1_SEQUENCE',
        output_path='tutorial_results/s22a1/data/reference_embedding.h5',
        # model_name='facebook/esm2_t33_650M_UR50D',
        model_name='facebook/esm2_t6_8M_UR50D',
        use_pooling=True,
        variant_id=1,  # Explicitly set variant_id to an integer
        device=None    # Explicitly set device to None to let the function determine the device
    )
    print('Reference sequence successfully embedded')
except Exception as e:
    print(f'Error embedding reference sequence: {e}')
    import sys
    sys.exit(1)
" || { echo "Failed to embed reference sequence. Exiting."; exit 1; }
echo "Reference sequence embedded to tutorial_results/s22a1/data/reference_embedding.h5"

# Step 5: Generate initial variants (Round 0)
echo -e "\nStep 5: Generating initial variants (Round 0)"
python -c "
from plm_framework.cli import propose_initial
propose_initial(
    config_path='tutorial_results/s22a1/config.yaml',
    sequence='$S22A1_SEQUENCE',
    output_path='tutorial_results/s22a1/proposals/round0_proposals.csv',
    batch_size=20,
    n_mutations=1,
    strategy='esm_logit',
    mutation_range=None,
)
"
echo "Initial variants generated at tutorial_results/s22a1/proposals/round0_proposals.csv"

# Step 6: Simulate experimental data for Round 0
echo -e "\nStep 6: Simulating experimental data for Round 0"
# In a real scenario, you would perform experiments on the proposed variants
# Here, we'll extract data from the Protein Gym dataset for the proposed variants
python -c "
import pandas as pd
import numpy as np

# Load proposed variants
proposals = pd.read_csv('tutorial_results/s22a1/proposals/round0_proposals.csv')
print(f'Loaded {len(proposals)} proposed variants')

# Load Protein Gym dataset
pg_data = pd.read_csv('tutorial_results/s22a1/data/S22A1_HUMAN_Yee_2023_activity.csv')
print(f'Loaded {len(pg_data)} Protein Gym variants')

# Extract mutant information from proposals
proposals['mutant'] = proposals['name']  # The name is already the mutation ID

# Merge with Protein Gym data to get scores
merged = proposals.merge(pg_data[['mutant', 'DMS_score']], on='mutant', how='left')

# For variants not in the dataset, print their IDs and drop them
missing_mask = merged['DMS_score'].isna()
if missing_mask.sum() > 0:
    missing_variants = merged.loc[missing_mask, 'mutant'].tolist()
    print(f'Dropping {missing_mask.sum()} variants not found in dataset:')
    print(f'Missing variants: {missing_variants}')
    merged = merged.dropna(subset=['DMS_score'])
    print(f'Remaining variants after dropping: {len(merged)}')

# Create assay results file - use id column which now contains mutation IDs
assay_results = merged[['id', 'sequence', 'DMS_score']].rename(columns={'DMS_score': 'score'})
assay_results['uncertainty'] = 0.1  # Assign fixed uncertainty

# Save to CSV
assay_results.to_csv('tutorial_results/s22a1/data/round0_results.csv', index=False)
print(f'Saved {len(assay_results)} assay results to tutorial_results/s22a1/data/round0_results.csv')
"
echo "Simulated experimental data saved to tutorial_results/s22a1/data/round0_results.csv"

# Step 7: Embed the measured variants
echo -e "\nStep 7: Embedding the measured variants"
python -c "
from plm_framework.cli import embed
embed(
    input_path='tutorial_results/s22a1/data/round0_results.csv',
    output_path='tutorial_results/s22a1/data/embeddings.h5',
    # model_name='facebook/esm2_t33_650M_UR50D',
    model_name='facebook/esm2_t6_8M_UR50D',
    batch_size=8,
    use_pooling=True
)
"
echo "Measured variants embedded to tutorial_results/s22a1/data/embeddings.h5"

# Step 8: Train initial model and run active learning loop
echo -e "\nStep 8: Training initial model and running active learning loop"
python -c "
from plm_framework.cli import learn
learn(
    config_path='tutorial_results/s22a1/config.yaml',
    input_path='tutorial_results/s22a1/data/round0_results.csv',
    output_dir='tutorial_results/s22a1/results',
    n_rounds=5,  # Increased from 2 to 5 rounds
    batch_size=10,
    strategy='ucb',
    temperature=1.0,
    clear_db=True  # Clear the database before starting
)
"
echo "Active learning completed, results saved to tutorial_results/s22a1/results"

# Step 9: Evaluate model on test data
echo -e "\nStep 9: Evaluating model on test data"
python -c "
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import torch
import os
import traceback

try:
    # Load Protein Gym dataset
    pg_data = pd.read_csv('tutorial_results/s22a1/data/S22A1_HUMAN_Yee_2023_activity.csv')

    # Split into train/test
    train_data, test_data = train_test_split(pg_data, test_size=0.2, random_state=42)
    print(f'Split dataset into {len(train_data)} training and {len(test_data)} test variants')

    # Save test data
    test_data.to_csv('tutorial_results/s22a1/data/test_data.csv', index=False)

    # Check if predictions file exists
    predictions_file = 'tutorial_results/s22a1/results/predictions.csv'
    
    if not os.path.exists(predictions_file):
        print(f'Predictions file not found: {predictions_file}')
        print('Generating predictions using the trained model...')
        
        # Load the trained model
        import pickle
        from plm_framework.embedding import ProteinEmbedder
        
        # Load the model
        model_path = 'tutorial_results/s22a1/results/model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Create embedder
        embedder = ProteinEmbedder(
            # model_name='facebook/esm2_t33_650M_UR50D',
            model_name = 'facebook/esm2_t6_8M_UR50D',
            use_pooling=True
        )
        
        # Embed test sequences
        test_sequences = test_data['mutated_sequence'].tolist()
        test_embeddings = []
        
        for seq in test_sequences:
            # Tokenize and embed
            inputs = embedder.tokenizer(seq, return_tensors='pt').to(embedder.device)
            with torch.no_grad():
                outputs = embedder.model(**inputs)
            
            # Get embeddings (use mean pooling if enabled)
            if embedder.use_pooling:
                # Mean pooling (excluding special tokens)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                
                # Create mask for non-special tokens (exclude first and last token)
                mask = attention_mask.clone()
                mask[:, 0] = 0  # Exclude first token (CLS)
                mask[:, -1] = 0  # Exclude last token (EOS)
                
                # Apply mask and compute mean
                masked_embeddings = embeddings * mask.unsqueeze(-1)
                sum_embeddings = masked_embeddings.sum(dim=1)
                sum_mask = mask.sum(dim=1).unsqueeze(-1)
                embedding = sum_embeddings / sum_mask
            else:
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0]
            
            test_embeddings.append(embedding.cpu().numpy().flatten())
        
        test_embeddings = np.array(test_embeddings)
        
        # Get predictions
        predictions = model.predict(test_embeddings)
        
        # Handle different return types
        if isinstance(predictions, tuple):
            test_predictions = predictions[0]
        else:
            test_predictions = predictions
            
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'mutant': test_data['mutant'],
            'predicted_score': test_predictions
        })
        
        # Save predictions
        os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
        predictions_df.to_csv(predictions_file, index=False)
        print(f'Generated and saved predictions to {predictions_file}')
    
    # Now load the predictions file
    predictions = pd.read_csv(predictions_file)
    
    # Merge with test data
    test_with_preds = test_data.merge(
        predictions[['mutant', 'predicted_score']], 
        on='mutant', 
        how='inner'
    )
    
    # Calculate metrics
    r2 = r2_score(test_with_preds['DMS_score'], test_with_preds['predicted_score'])
    rmse = np.sqrt(mean_squared_error(test_with_preds['DMS_score'], test_with_preds['predicted_score']))
    spearman, _ = spearmanr(test_with_preds['DMS_score'], test_with_preds['predicted_score'])
    
    print(f'Model evaluation on {len(test_with_preds)} test variants:')
    print(f'R² score: {r2:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Spearman correlation: {spearman:.4f}')
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(test_with_preds['DMS_score'], test_with_preds['predicted_score'], alpha=0.6)
    plt.xlabel('Actual DMS Score')
    plt.ylabel('Predicted Score')
    plt.title('Model Predictions vs Actual Values')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add diagonal line
    min_val = min(test_with_preds['DMS_score'].min(), test_with_preds['predicted_score'].min())
    max_val = max(test_with_preds['DMS_score'].max(), test_with_preds['predicted_score'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add metrics as text
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\\nRMSE = {rmse:.4f}\\nSpearman = {spearman:.4f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Create plots directory if it doesn't exist
    os.makedirs('tutorial_results/s22a1/plots', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('tutorial_results/s22a1/plots/model_evaluation.png', dpi=300)
    print('Evaluation plot saved to tutorial_results/s22a1/plots/model_evaluation.png')
except Exception as e:
    print(f'Error evaluating model: {e}')
    traceback.print_exc()
"

# Step 10: Propose new variants for next round
echo -e "\nStep 10: Proposing new variants for next round"
python -c "
from plm_framework.cli import propose
import pandas as pd

# Generate candidate sequences (in a real scenario, these would be designed)
# Here we'll use variants from the Protein Gym dataset that weren't used in training
try:
    # Load Protein Gym dataset
    pg_data = pd.read_csv('tutorial_results/s22a1/data/S22A1_HUMAN_Yee_2023_activity.csv')
    
    # Load variants used in training
    used_variants = pd.read_csv('tutorial_results/s22a1/results/variants.csv')
    
    # Find unused variants
    used_mutants = set(used_variants['name'].tolist())
    unused_variants = pg_data[~pg_data['mutant'].isin(used_mutants)]
    
    # Save as candidates file
    candidates = unused_variants[['mutant', 'mutated_sequence']].rename(
        columns={'mutant': 'name', 'mutated_sequence': 'sequence'}
    )
    candidates['id'] = range(1, len(candidates) + 1)
    candidates = candidates[['id', 'name', 'sequence']]
    candidates.to_csv('tutorial_results/s22a1/data/candidates.csv', index=False)
    print(f'Saved {len(candidates)} candidate variants to tutorial_results/s22a1/data/candidates.csv')
    
    # Propose variants using trained model
    from plm_framework.cli import propose
    propose(
        config_path='tutorial_results/s22a1/config.yaml',
        model_path='tutorial_results/s22a1/results/model.pkl',
        candidates_path='tutorial_results/s22a1/data/candidates.csv',
        output_path='tutorial_results/s22a1/proposals/next_round_proposals.csv',
        batch_size=20,
        strategy='ucb',
        temperature=1.0
    )
    print('Proposed variants for next round saved to tutorial_results/s22a1/proposals/next_round_proposals.csv')
except Exception as e:
    print(f'Error proposing new variants: {e}')
"

echo -e "\nTutorial completed! Results are in the tutorial_results/s22a1 directory."
echo "To continue the active learning loop, you would:"
echo "1. Perform experiments on the proposed variants"
echo "2. Update the assay results file with new measurements"
echo "3. Run the learn command again to update the model and propose new variants"
