from transformers import pipeline
import torch

fin_analyzer = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

def generate_insights(text):
    prompt = f"Analyze this document and identify 3 key risks and 2 opportunities:\n{text}"
    result = fin_analyzer(prompt, max_length=500, num_return_sequences=1)
    return result[0]["generated_text"]