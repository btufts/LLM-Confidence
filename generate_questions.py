import torch
import gc
from llm.models import *
from data.dataset import *
from tqdm import tqdm
import json

def main():
  torch.cuda.empty_cache()
  gc.collect()
  model = Mistral()

  data = SVAMPloader("data/SVAMP.json")
  amount = 100
  answers = []
  for q,a,w in tqdm(zip(data.questions[:amount], data.answers[:amount], data.wrong[:amount])):
    d = {}
    d["question"] = q
    d["answer"] = a
    d["wrong"] = w
    new_questions = []

    for _ in range(3):
      messages = [
        {"role": "user", "content": f"""Rephrase the following question such that the answer remains the same. Include all details of the question. **Respond only with the reformatted question.**
        Question: {q}\nReformatted Question:"""},
      ]
      response = model.generate(messages)
      question = response[response.find("[/INST]")+8:response.find("</s>")]
      new_questions.append(question)

    d["new_questions"] = new_questions
    answers.append(d)


  with open("data/augmented_SVAMP.json", 'w') as json_file:
    json.dump(answers, json_file)

if __name__ == "__main__":
  main()