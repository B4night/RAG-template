from datasets import load_from_disk
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.nn.functional import cosine_similarity

DPR_DIM = 768

ds = load_from_disk('./nqc_data')
# print(ds)

text_splitter = CharacterTextSplitter(
    separator="", #This will split the text by characters
    chunk_size=200, #Number of characters in each chunk 
    chunk_overlap=50, #Number of overlapping characters between chunks
)

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("../model/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("../model/dpr-question_encoder-single-nq-base")
model.eval()

client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name="nq"):
    client.drop_collection(collection_name="nq")

client.create_collection(
    collection_name="nq",
    dimension=DPR_DIM,
)

cnt = 0
for example in ds['train']:
    id = example['id']
    title = example['title']
    question = example['question']
    long_answer = example['long_answers']
    short_answer = example['short_answers']
    
    chunks=text_splitter.create_documents([long_answer[0]])
    chunk_embeddings = []
    
    chunk_combinations = []
    
    for chunk in chunks:
        if len(chunk.page_content) < 100:
            chunk_combinations.append(chunk.page_content)
            
            length = 0
            for c in chunk_combinations:
                length += len(c)
            if length > 150:
                chunk.page_content = " ".join(chunk_combinations)
                chunk_combinations = []
            else:
                continue
        input_ids = tokenizer(chunk.page_content, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output.squeeze(dim=0)
        chunk_embeddings.append(embeddings)
    
    data = [
        {"id": int(id), "title": title, "question": question, "context": chunks[i].page_content, "vector": chunk_embeddings[i].detach().numpy().tolist()}
        for i in range(len(chunk_embeddings))
    ]
    res = client.insert(collection_name="nq", data=data)
    print(res)
    
    cnt += 1
    if cnt > 10:
        break