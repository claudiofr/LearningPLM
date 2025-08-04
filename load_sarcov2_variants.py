import pandas as pd
from Bio import SeqIO

def load_sarscov2_variants(csv_path, fasta_path):
    # Load wildtype sequence
    wt_seq = str(next(SeqIO.parse(fasta_path, "fasta")).seq)

    # Load mutation data
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        pos = int(row["site"]) - 1  # 0-based index
        wt = row["wildtype"]
        mut = row["mutation"]
        if wt_seq[pos] != wt:
            continue  # skip inconsistent data
        mut_seq = wt_seq[:pos] + mut + wt_seq[pos + 1:]
        variant_id = f"{wt}{pos+1}{mut}"
        score = row["mut_escape"]
        rows.append({
            "id": variant_id,
            "sequence": mut_seq,
            "score": score,
            "uncertainty": 0.01  # dummy uncertainty
        })

    # Save to formatted CSV
    pd.DataFrame(rows).to_csv("data/sarscov2_dataset.csv", index=False)

    # Return structured data if needed later
    return rows, wt_seq

if __name__ == "__main__":
    variants, wt_seq = load_sarscov2_variants(
        csv_path="data/sarscov2_preprocessed.csv",
        fasta_path="data/cov2_S_WT.fasta"
    )

    print(f"Loaded {len(variants)} variants.")
    print("First 3 variants:")
    for v in variants[:3]:
        print(f"  ID: {v['id']} | Score: {v['score']} | Seq (start): {v['sequence'][:10]}...")
