# change the vocab of numbers to a new vocab
# e.g. {Q: 1, A: 2} to {Q: 'apple', A: 'banana'}
# Usage: python change_vocab.py old_text_label.json new_text_label.json
# change the script if the data structure is different


import sys
import json


def main():
    old_text_path = sys.argv[1]
    new_text_path = sys.argv[2]

    num_to_vocab = {}
    num_threshold = 30000         
    with open('vocab.txt', 'r') as file:
        for line_number, line_content in enumerate(file, 1):
            line_content = line_content.strip()
            if line_number>=(num_threshold-1000):
                num_to_vocab[line_number] = line_content
    with open(old_text_path, 'r') as file:
        old_text = json.load(file)
    new_text = {}
    for key, value in old_text.items():
        new_text[key] = num_to_vocab[value]
    with open(new_text_path, 'w') as file:
        json.dump(new_text, file, indent=4)

if __name__ == "__main__":
    main()