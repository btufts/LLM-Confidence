import json

class SelfAwareDataLoader:
    """
    DataLoader for the SelfAware dataset which includes both answerable and unanswerable questions.
    The DataLoader loads the dataset from a JSON file and provides methods to access
    the questions, answers (if available), and other metadata like answerability.
    """

    def __init__(self, path):
        self.path = path
        self.data = self.load_json()
        self.index = 0  

    def load_json(self):
        """
        Loads JSON data from the specified file path.
        """
        with open(self.path, 'r') as file:
            data = json.load(file)
        return data["example"]
    
    def __iter__(self):
        """
        Returns the iterator object (self).
        """
        return self

    def __next__(self):
        """
        Allows iteration over questions. Each call fetches the next question and its details.
        """
        if self.index < len(self.data):
            item = self.data[self.index]
            self.index += 1
            return {
                "question_id": item['question_id'],
                "question": item['question'],
                "answer": item.get('answer', None),  
                "answerable": item['answerable'],
                "source": item['source']
            }
        else:
            raise StopIteration
    
    def reset(self):
        """
        Resets the dataset index for fresh iteration.
        """
        self.index = 0


def main():
    loader = SelfAwareDataLoader("data/SelfAware.json")
    
    for item in loader:
        print(f"Question ID: {item['question_id']}, Question: {item['question']}, Answer: {item['answer']}, Answerable: {item['answerable']}, Source: {item['source']}")

    loader.reset()  # Reset the iterator if you need to go through the data again

if __name__ == "__main__":
    main()
