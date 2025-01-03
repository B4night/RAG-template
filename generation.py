from datasets import load_from_disk
from langchain.text_splitter import CharacterTextSplitter
from pymilvus import MilvusClient
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.nn.functional import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("./model/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("./model/dpr-question_encoder-single-nq-base")
model.eval()

client = MilvusClient("./data/milvus_demo.db")

query = "Who won the 2023 Cricket World Cup?"

query_emb = model(tokenizer(query, return_tensors="pt")["input_ids"]).pooler_output.squeeze(dim=0).tolist()

res = client.search(collection_name="nq", 
                    data=[query_emb], 
                    limit=5, 
                    output_fields=["context", "vector"])

contexts = []

for i, hits in enumerate(res):
    for j, hit in enumerate(hits):
        contexts.append(hit['entity']['context'])



model_id = "./model/Meta-Llama-3-8B-Instruct"

# 1. 加载分词器与模型
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # 与原先pipeline保持一致
    device_map="auto"            # 与原先pipeline保持一致
)

context_str = " ".join(contexts)
print(f'context_str: {context_str}')

augmented_prompt = f"\
    Context: {context_str}\
    Question: {query}\
    Above is a context and a question. Please generate an answer to the question based on the context. If the context is not needed to generate the answer, please ignore it."

# 2. 准备输入
# augmented_prompt = "Who won the 2023 Cricket World Cup?"
input_ids = tokenizer(augmented_prompt, return_tensors="pt").input_ids.to(model.device)

# 3. 执行生成
# 可根据需要调整参数，如 max_new_tokens, temperature, etc.
output_ids = model.generate(
    input_ids,
    max_new_tokens=128,  # 生成文本的长度可自定义
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

# 4. 解码输出
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 5. 打印或返回结果
print("\n\nGenerated text:")
print(generated_text)
