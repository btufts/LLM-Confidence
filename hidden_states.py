import torch
import gc
from llm.models import *
from data.dataset import *
from tqdm import tqdm
import json
import torch.nn.functional as F

def cosine_similarity(tensor1, tensor2):
    max_len = max(tensor1.size(-1), tensor2.size(-1))

    # Calculate the difference in lengths
    diff1 = max_len - tensor1.size(1)
    diff2 = max_len - tensor2.size(1)
    
    # Pad the shorter tensor with zero vectors along the first dimension
    if diff1 > 0:
        pad1 = torch.zeros(tensor1.size(0), diff1, tensor1.size(2))
        tensor1 = torch.cat((tensor1, pad1), dim=1)
    if diff2 > 0:
        pad2 = torch.zeros(tensor2.size(0), diff2, tensor2.size(2))
        tensor2 = torch.cat((tensor2, pad2), dim=1)

    # tensor1_padded = F.pad(tensor1, pad=(0, max_len - tensor1.size(-1)))
    # tensor2_padded = F.pad(tensor2, pad=(0, max_len - tensor2.size(-1)))
    # Normalize the input tensors
    tensor1_normalized = F.normalize(tensor1, p=2, dim=-1)
    tensor2_normalized = F.normalize(tensor2, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = torch.sum(tensor1_normalized * tensor2_normalized, dim=-1)
    similarity = torch.mean(similarity, dim=-1)
    
    return similarity.item()  # Convert to Python float

def main():
  torch.cuda.empty_cache()
  gc.collect()
  model = Mistral()

  data = SVAMPloader("data/SVAMP.json")
  amount = 1000
  answers = []
  for q,a,w in tqdm(zip(data.questions[:amount], data.answers[:amount], data.wrong[:amount])):
    d = {}
    d["question"] = q
    d["answer"] = a
    d["wrong"] = w
    messages = [
        {"role": "user", "content": f"""Read the question, provide your answer and your explanation for this answer.
      Use the following format in **all of your responses**:
      ```Answer and Explanation: ([ONLY the number], [Your Breif Explanation])```
      Only respond with the answer and explanation, nothing else.
      Question: {d["question"]}
      Now, please answer this question and provide your explanation."""},
      ]
    response = model.generate(messages)
    answer = response[response.find("[/INST]")+8:response.find("</s>")]
    answer = answer[answer.find(":"):]
    answer = answer[answer.find("(")+1:answer.find(")")]
    d["first_answer"] = answer
    og_response = model.get_hidden(answer).cpu()
    answers.append(d)

  with open("data/explanationresponses.json", 'w') as json_file:
    json.dump(answers, json_file)

if __name__ == "__main__":
  main()