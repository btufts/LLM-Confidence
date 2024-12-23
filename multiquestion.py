import torch
import gc
from llm.models import *
from data.dataset import *
from tqdm import tqdm
import json
import torch.nn.functional as F
import numpy as np

def cosine_similarity(A, B):
    A = A[0,:,:]
    B = B[0,:,:]

    if A.shape[0] < B.shape[0]:
        A = np.vstack((A, np.ones((B.shape[0] - A.shape[0], A.shape[1]))))
    elif B.shape[0] < A.shape[0]:
        B = np.vstack((B, np.ones((A.shape[0] - B.shape[0], B.shape[1]))))

    dot_product = np.sum(A * B, axis=0)

    # Compute magnitudes
    magnitude_A = np.sqrt(np.sum(A * A, axis=0))
    magnitude_B = np.sqrt(np.sum(B * B, axis=0))

    # Compute cosine similarity
    cosine_sim = np.mean(dot_product / (magnitude_A * magnitude_B))
    return cosine_sim


def main():
  torch.cuda.empty_cache()
  gc.collect()
  model = Mistral()

  with open("data/augmented_SVAMP.json", 'r') as json_file:
    data = json.load(json_file)
  answers = []
  for d in tqdm(data):
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
    og_response = model.get_hidden(answer).cpu().numpy()
    sim = 0
    sub_answers = []
    for qs in d["new_questions"]:
      messages = [
        {"role": "user", "content": f"""Read the question, provide your answer and your explanation for this answer.
      Use the following format in **all of your responses**:
      ```Answer and Explanation: ([ONLY the number], [Your Brief Explanation])```
      Only respond with the answer and explanation, nothing else.
      Question: {qs}
      Now, please answer this question and provide your explanation."""},
      ]
      response = model.generate(messages)
      answer = response[response.find("[/INST]")+8:response.find("</s>")]
      answer = answer[answer.find(":"):]
      answer = answer[answer.find("(")+1:answer.find(")")]
      sub_answers.append(answer)
      response = model.get_hidden(answer).cpu().numpy()
      sim += cosine_similarity(og_response, response)

    d["avg_sim"] = sim / len(d["new_questions"])
    print(d["avg_sim"])
    d["sub_answers"] = sub_answers
    answers.append(d)

  with open("data/multiquestionresponses.json", 'w') as json_file:
    json.dump(answers, json_file)

if __name__ == "__main__":
  main()