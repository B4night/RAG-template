import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.nn.functional import cosine_similarity

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("./model/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("./model/dpr-question_encoder-single-nq-base")
model.eval()  # 设置模型为评估模式

with torch.no_grad():
    while True:
        query = input("Please input a question: ")
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output

        # Test similarity for similar questions
        query = input("Please input a similar question: ")
        query_embedding = model(tokenizer(query, return_tensors="pt")["input_ids"]).pooler_output

        similarity = cosine_similarity(embeddings, query_embedding).item()
        print(f"Cosine Similarity: {similarity}")
