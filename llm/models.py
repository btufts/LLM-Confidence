from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.model import get_conversation_template
import torch
import os
import gc

class HuggingFace():
  def __init__(self) -> None:
    self.model = None
    self.tokenizer = None

class Mistral(HuggingFace):
  def __init__(self, model_path = "mistralai/Mistral-7B-Instruct-v0.2", tokenizer_path = "mistralai/Mistral-7B-Instruct-v0.2", device = "cuda") -> None:
    super(HuggingFace).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,token=os.getenv("HUGGING_FACE_TOKEN"), load_in_4bit=True)
    self.model = AutoModelForCausalLM.from_pretrained(model_path,token=os.getenv("HUGGING_FACE_TOKEN"), device_map=device, load_in_4bit=True)
    self.device = device
    # self.eos_token_ids = [self.tokenizer.encode("\n")[1], self.tokenizer.encode(".")[1]]
    self.stop_tokens = self.tokenizer.convert_tokens_to_ids(["<EOS>", "<END>", "\n"])

  def generate(self, 
                  messages,
                  max_n_tokens: int = 200, 
                  temperature: float = 0.7,
                  top_p: float = 1.0):
    # Messages is list of dictionaries like : {"role": "user", "content", "Hello"}
    encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
    # encodeds = self.tokenizer.encode(messages, return_tensors="pt")
    encodeds = encodeds.to(self.device)

    output_ids = self.model.generate(
                encodeds,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.stop_tokens
            ) # send back to cpu in case we were using a gpu

    decoded = self.tokenizer.batch_decode(output_ids)

    return decoded[0]

  def get_hidden(self, answer):
    token_ids = self.tokenizer([answer], return_tensors="pt").to(self.device)

    # Inference
    with torch.no_grad():
        outputs = self.model(**token_ids, output_hidden_states=True)

    # Retrieve the hidden states
    hidden_states = outputs.hidden_states

    # Extract the final hidden state
    final_hidden_state = hidden_states[-1]  # Assuming the last layer is used

    return final_hidden_state

class Llama(HuggingFace):
  def __init__(self, model_path = "meta-llama/Llama-2-13b-chat-hf", tokenizer_path = "meta-llama/Llama-2-13b-chat-hf", device = "cuda") -> None:
    super(HuggingFace).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,token=os.getenv("HUGGING_FACE_TOKEN"))
    self.model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                      token=os.getenv("HUGGING_FACE_TOKEN"),
                                                      device_map="auto",
                                                      torch_dtype=torch.float16)
    self.device = device
    # self.eos_token_ids = [self.tokenizer.encode("\n")[1], self.tokenizer.encode(".")[1]]
    self.stop_tokens = self.tokenizer.convert_tokens_to_ids(["<EOS>", "<END>", ".", "\n", "</s>"])

  def get_conversation(self):
    template = get_conversation_template("llama-2")
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template

  def generate(self, 
                  prompt,
                  max_n_tokens: int = 100, 
                  temperature: float = 0.7,
                  top_p: float = 1.0):
    # List of dictionaries like : {"role": "user", "content", "Hello"}
    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.stop_tokens
            ).cpu() # send back to cpu in case we were using a gpu

    output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output

