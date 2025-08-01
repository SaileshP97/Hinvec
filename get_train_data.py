import json
import os
import glob
import random

def get_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for sample in f:
            data.append(json.loads(sample))
    return data

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')


def main():
    train_data = []
    val_data = []

    files = glob.glob("./training_data/*/train.jsonl", recursive=True,)
    for file in files:
        train_data.extend(get_jsonl(file))
    random.shuffle(train_data)
    save_jsonl(train_data, "./training_data/train_data.jsonl")

    files = glob.glob("./training_data/*/val.jsonl", recursive=True,)
    for file in files:
        val_data.extend(get_jsonl(file))
    random.shuffle(val_data)
    save_jsonl(val_data, "./training_data/val_data.jsonl")
        
    
if __name__ == "__main__":
    main()