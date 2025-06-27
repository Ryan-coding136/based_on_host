from Bio import SeqIO
import pandas as pd


train_ids = set(pd.read_csv("train_metadata.csv")["Isolate_Id"])
test_ids = set(pd.read_csv("test_metadata.csv")["Isolate_Id"])


input_fasta = "H1_H3_H5_H7_H9_aligned.fasta"  
train_out = "train_sequences.fasta"
test_out = "test_sequences.fasta"

with open(train_out, "w") as train_f, open(test_out, "w") as test_f:
    for record in SeqIO.parse(input_fasta, "fasta"):
        header = record.id
        try:
            epi_id = header.split("|")[1]  # 假设格式为 A_name|EPI_ISL_xxx|...
        except IndexError:
            print(f"⚠️ Unable to analyze header: {header}")
            continue

        if epi_id in train_ids:
            SeqIO.write(record, train_f, "fasta")
        elif epi_id in test_ids:
            SeqIO.write(record, test_f, "fasta")
