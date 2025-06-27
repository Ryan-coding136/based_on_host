import torch
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

# 设置路径
fasta_path = "split9/H9_sequences.fasta"
output_path = "split9/H9_sequences.npy"

# 加载 ESM-2 模型（1280 维）
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()  # 使用 GPU

# 解析 FASTA 序列
sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]

# 批处理设置
batch_size = 8  # 可根据你的显存调节
all_embeddings = []

# 处理并生成嵌入
for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
    batch = sequences[i:i + batch_size]
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.cuda()

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    representations = results["representations"][33]

    for j, (_, seq) in enumerate(batch):
        seq_len = len(seq)
        embedding = representations[j, 1:seq_len + 1].cpu().numpy()
        # 平均池化为 (1280,) 向量
        avg_embedding = embedding.mean(axis=0)
        all_embeddings.append(avg_embedding)

# 保存为 numpy 文件
np.save(output_path, np.array(all_embeddings))
print(f"✅ Saved to: {output_path} | Shape: {np.array(all_embeddings).shape}")
