import json
import random

class SVAMPloader:
    """
        Here is a dataloader for the SVAMP dataset of math word problems. To use it, call it with the
        path to the SVAMP.json file. From that the relevant fields are

        questions: a list of all word problems

        answers: a list of all correct answers

        wrong: a list of incorrect followups generated

        Each is in the format of a list, where each field corresponds to the same index. For example, 
        taking the 0th index of each you should get

        question: Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack, 
                  how much do you have to pay to buy each pack?
        
        answer: 51.0

        wrong: No, that's wrong. The correct answer is 271.35 (this will be random)

        The main function illustrates how to use this dataloader.
    """
    def __init__(self, path):
        self.path = path
        self.json = self.load_json()
        self.answers = self.get_answers()
        self.questions = self.get_questions()
        self.wrong = self.get_followups()
    
    def load_json(self):
        with open(self.path, 'r') as f:
            data = json.load(f)  
        return data
    
    def get_answers(self):
        return [item["Answer"] for item in self.json]
    
    def get_questions(self):
        questions = []
        for item in self.json:
            b = item['Body'] + ', '
            q = item['Question'][0].lower() + item['Question'][1:]
            questions.append(b+q)
        return questions
    
    def get_followups(self):
        followups = []
        for ans in self.answers:
            while True:
                wrong = round(random.uniform(0, 1000), 2)
                if ans != wrong:
                    break
            followup = f"No, that's wrong. The correct answer is {wrong}."
            followups.append(followup)
        return followups
    

class SQUADLoader:
    """
        Here is a dataloader for the SQUAD v2.0 dataset of question/answer data. To use it, call it with the
        path to the SVAMP.json file. From that the relevant fields are

        questions: a list of all questions

        answers: a list of all correct answers

        wrong: a list of incorrect followups generated

        Each is in the format of a list, where each field corresponds to the same index. 
        For wrong, I just had it choose another random answer from the list of answers,
        though this may be too nonsensical for some answers to be ideal.

        The main function illustrates how to use this dataloader.
    """
    def __init__(self, path):
        self.path = path
        self.json = self.load_json()
        self.qa = self.get_qa()
        self.answers = self.get_answers()
        self.questions = self.get_questions()
        self.wrong = self.get_followups()
    
    def load_json(self):
        with open(self.path, 'r') as f:
            data = json.load(f)  
        return data
    
    def get_qa(self):
        question_answer = []
        for item in self.json["data"]:
            for paragraph in item["paragraphs"]:
                for qa in paragraph["qas"]:
                    if qa["answers"]:
                        question = qa["question"]
                        answer = qa["answers"][0]["text"]  
                        question_answer.append((question, answer))
                    else:
                        continue
        return question_answer

    def get_answers(self):
        return [item[1] for item in self.qa]
    
    def get_questions(self):
        questions = [item[0] for item in self.qa]
        self.qa = None
        return questions
    
    def get_followups(self):
        followups = []
        max_index = len(self.answers) - 1
        for ans in self.answers:
            while True:
                id = random.randint(0, max_index)
                wrong = self.answers[id]
                if ans != wrong:
                    break
            followup = f"No, that's wrong. The correct answer is {wrong}."
            followups.append(followup)
        return followups
    
def main():
    print("Here is the SVAMP dataset: ")
    data = SVAMPloader("SVAMP.json")
    print(data.questions[0])
    print(data.answers[0])
    print(data.wrong[0])
    print(len(data.questions))
    print(len(data.answers))
    print(len(data.wrong))
    print("\nNow for the SQUAD dataset: ")
    data = SQUADLoader("train-v2.0.json")
    print(data.questions[0])
    print(data.answers[0])
    print(data.wrong[0])
    print(len(data.questions))
    print(len(data.answers))
    print(len(data.wrong))

if __name__ == "__main__":
    main()