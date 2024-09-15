from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-1.1-7b-it"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The meaning of life is"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

output = model.generate(input_ids, max_length=100)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)