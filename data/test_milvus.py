from datasets import load_from_disk
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.nn.functional import cosine_similarity


tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("../model/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("../model/dpr-question_encoder-single-nq-base")
model.eval()

client = MilvusClient("milvus_demo.db")

query = "Who won the 2023 Cricket World Cup?"

query_emb = model(tokenizer(query, return_tensors="pt")["input_ids"]).pooler_output.squeeze(dim=0).tolist()

res = client.search(collection_name="nq", 
                    data=[query_emb], 
                    limit=5, 
                    output_fields=["question", "context", "vector"])

for i, hits in enumerate(res):
    print(f"Search results for query {i}:")
    
    # hits 本身又是一个列表 (或 ExtraList)，里面有每一条命中的结果
    for j, hit in enumerate(hits):
        # hit 有以下常见属性:
        # - hit.id: 主键
        # - hit.distance: 与查询向量的距离 (或相似度 score)
        # - hit.entity.get("<field_name>"): 获取额外输出字段的值
        print(f"  Hit {j}:")
        print(f"    id = {hit['id']}")
        print(f"    distance = {hit['distance']}")
        
        print(f"    question = {hit['entity']['question']}")
        print(f"    context = {hit['entity']['context']}")

    print("-" * 50)