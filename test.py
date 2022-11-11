from transformers import T5Tokenizer, AutoModelForCausalLM
import torch

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

prompt = "ゆっくり霊夢「ゆっくり霊夢です。」ゆっくり魔理沙「ゆっくり魔理沙だぜ。」"
num_return_sequences = 10

input_ids = tokenizer.encode(prompt, return_tensors="pt",add_special_tokens=False).to(device)
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num_return_sequences
    )
decoded = tokenizer.batch_decode(output,skip_special_tokens=False)
f = open('output.txt', 'w', encoding='UTF-8')
for i in range(num_return_sequences):
  print(decoded[i])
  f.write(decoded[i]+"\n")
f.close()