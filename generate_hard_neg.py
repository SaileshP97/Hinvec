from openai import OpenAI
import random
import time
import re
import json
import loguru

loguru.logger.add("logs/generate_hard_neg.log", rotation="1 MB", retention="10 days", level="DEBUG")
loguru.logger.info("Starting hard negative generation...")

def generate_query_for_article(sample):

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )
    
    system_message = """
        You are an AI assistant designed to generate challenging hard negative examples in the same language as the output. Your task is to produce exactly one concise and well-formed hard negative response that seems similar to the correct Output text, but is actually irrelevant for the given Input text. The hard negative should be misleading in a subtle way — close in topic or style, but not a valid answer. Make sure the grammar and vocabulary are correct. Wrap the hard negative inside ## markers like this: ## hard negative text ##*.**
    """
    
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.3-nemotron-super-49b-v1",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Input Text: {sample['source']} \nOutput Text: {sample['target']} \nGenerate Hard Negative example: "}
            ],
            temperature=0.6,
            top_p=0.95,
            max_tokens=100,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False
        )
    
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
        return generate_query_for_article(sample)
    
    try:
        return None, completion.choices[0].message.content.split("##")[1].strip()
    except IndexError:
        return 1, completion.choices[0].message.content.strip()


if __name__ == "__main__":
    loguru.logger.info("Loading training data...")
    
    try:
        with open("./new_training_data/train_data_with_hard_negative.jsonl", "r") as f:
            training_data = []
            for sample in f:
                training_data.append(json.loads(sample))
        loguru.logger.info("Loaded training data with hard negatives.")
    except FileNotFoundError:
        loguru.logger.info("No existing hard negatives found, loading original training data.")
        loguru.logger.info("Loading original training data...")
        with open("./new_training_data/train_data.jsonl", "r") as f:
            training_data = []
            for sample in f:
                training_data.append(json.loads(sample))

    for idx, sample in enumerate(training_data):

        if sample.get('hard_negative_flag', 0) == 1:
            continue
        flag, training_data[idx]['hard_negative'] = generate_query_for_article(sample)
        training_data[idx]['hard_negative_flag'] = 1
        if flag == 1:
            training_data[idx]['hard_negative_##'] = 1

        if idx%5000==0:
            loguru.logger.info(f"Processed {idx} samples, current sample: {sample['source']}")
            with open("./new_training_data/train_data_with_hard_negative.jsonl", "w") as f:
                for sample in training_data:
                    json.dump(sample, f, ensure_ascii=False)
                    f.write('\n')

    loguru.logger.info("Generation complete, saving final data...")
    with open("./new_training_data/train_data_with_hard_negative.jsonl", "w") as f:
        for sample in training_data:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    loguru.logger.info("Final data saved successfully.")
    loguru.logger.info("Hard negative generation completed.")