from Bio import SeqIO

def extract_ids_from_fasta(fasta_file, output_id_file=None):
    """
    提取 fasta 文件中的 EPI_ISL ID 并保存（可选）
    """
    epi_ids = [record.id.split("|")[1] for record in SeqIO.parse(fasta_file, "fasta")]

    if output_id_file:
        with open(output_id_file, "w") as f:
            for eid in epi_ids:
                f.write(eid + "\n")
        print(f"✅ EPI IDs saved to {output_id_file}")

    return epi_ids


def compare_id_order(fasta_file, id_txt_file):
    """
    比较 fasta 中的 ID 顺序 和 txt 中的顺序是否一致
    """
    fasta_ids = extract_ids_from_fasta(fasta_file)
    
    with open(id_txt_file) as f:
        txt_ids = [line.strip() for line in f]
    
    if fasta_ids == txt_ids:
        print("✅ ")
    else:
        print("❌ ")

        for i, (fid, tid) in enumerate(zip(fasta_ids, txt_ids)):
            if fid != tid:
                print(f"{i}Unmatched: {fid} ≠ {tid}")
                break


# ✅
if __name__ == "__main__":
    fasta_path = "split5/H5_sequences.fasta"
    id_file = "split5/H5_ids.txt"


    extract_ids_from_fasta(fasta_path, output_id_file=id_file)


    compare_id_order(fasta_path, id_file)
