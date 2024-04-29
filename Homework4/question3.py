from transformers import AutoModelForCausalLM, AutoTokenizer
import time

prompt = "Write a poem about Spring."

model_id = "EleutherAI/pythia-160m"
assistant_model_id = "EleutherAI/pythia-14m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_model_id)


# Track the time of standard decoding
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=128, return_dict_in_generate=True)
end_time = time.time()
input_length = inputs.input_ids.shape[1]
token = outputs.sequences[0, input_length+1:] 
print(f"[INFO] raw token: {token}")
output = tokenizer.decode(token)
print(f"[Context]: {prompt} \n[Output]:{output}\n")
print(f"The standard decoding takes {end_time-start_time} seconds")


# Track the time of speculative decoding
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=128, assistant_model=assistant_model, return_dict_in_generate=True)
end_time = time.time()
input_length = inputs.input_ids.shape[1]
token = outputs.sequences[0, input_length+1:] 
print(f"[INFO] raw token: {token}")
output = tokenizer.decode(token)
print(f"[Context]: {prompt} \n[Output]:{output}\n")
print(f"The speculative decoding takes {end_time-start_time} seconds")
