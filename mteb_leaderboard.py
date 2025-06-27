import json
import os
import glob
import numpy as np
import csv

parent_dir = os.path.dirname(os.getcwd())
path = f"{parent_dir}/results/results/"

def create_model_scores_csv(data, output_file="model_scores.csv"):
    
    # Get all unique task names across all models
    all_tasks = set()
    for model_data in data.values():
        all_tasks.update(model_data.keys())
    
    # Remove any None values from tasks
    all_tasks = [task for task in all_tasks if task]
    
    # Sort tasks for consistent output
    all_tasks = sorted(all_tasks)
    
    # Prepare data for CSV
    csv_data = []
    
    for model_name, model_data in data.items():
        # Create a row for this model
        row = {'model_name': model_name}
        
        # Add scores for each task
        valid_scores = []
        for task in all_tasks:
            score = model_data.get(task)
            row[task] = score
            if score is not None:  # Only include non-None values in mean calculation
                valid_scores.append(score)
        
        # Calculate mean score
        if valid_scores:
            row['mean_score'] = np.mean(valid_scores)
        else:
            row['mean_score'] = None
        
        csv_data.append(row)
    
    # Sort models by mean score (descending)
    csv_data.sort(key=lambda x: x['mean_score'] if x['mean_score'] is not None else -1, reverse=True)
    
    # Define column order: model_name, mean_score, followed by task names
    columns = ['model_name', 'mean_score'] + all_tasks
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"CSV file created successfully: {output_file}")
    return output_file


def main():

    models = ['Alibaba-NLP__gte-Qwen2-7B-instruct','Linq-AI-Research__Linq-Embed-Mistral',
              'Cohere__Cohere-embed-multilingual-v3.0', 'OrdalieTech__Solon-embeddings-large-0.1',
              'Alibaba-NLP__gte-Qwen2-1.5B-instruct', 'Lajavaness__bilingual-embedding-large',
              'Salesforce__SFR-Embedding-2_R', 'intfloat__multilingual-e5-large-instruct', 
              'Ganga_org', 'Ganga_1024',
              'ganga-2-1b-embeddings-full-mean-32', 'ganga-2-1b-embeddings-new-equal-mean-32',
              'ganga-2-1b-embeddings-new-full-mean-32', 'ganga-2-1b-embeddings-new-full-mean-32-epoch-3',
              'ganga-2-1b-embeddings-new-equall-finetune-mean-32-epoch-1',
              'ganga-2-1b-embeddings-new-equall-finetune-2-mean-32-epoch-1',
              'ganga-2-1b-embeddings-new-equall-finetune-3-mean-32-epoch-1',
              'ganga-2-1b-embeddings-new-equall-finetune-final-2-mean-32-epoch-1',
              'ganga-2-1b-embeddings-new-equall-finetune-final-mean-32-epoch-1',
              'ganga-2-1b-embeddings-new-equall-bidir-mean-42-epoch-1',
              'ganga-2-1b-embeddings-new-equall-bidir-eos-42-epoch-1',
              'Sailesh97__Hinvec',
              'ganga-2-1b-embeddings-new-equall-eos-42-epoch-1']
    
    tasks = ["BelebeleRetrieval", "XQuADRetrieval", "IndicCrosslingualSTS",
             "WikipediaRerankingMultilingual", "XNLI", "SIB200ClusteringS2S",
             "TweetSentimentClassification", "MultiHateClassification", "MTOPIntentClassification",
             "IndicLangClassification", "SentimentAnalysisHindi", "HindiDiscourseClassification",
             "LinceMTBitextMining", "IndicGenBenchFloresBitextMining", "IN22GenBitextMining",
             "IN22ConvBitextMining"]
    
    model_report = {}
    
    for model in models:

        print(model)

        task_report = {}
        for task in tasks:

            if model == "Ganga_mean":
                taskss = f"results/Ganga_mean/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)

            elif model == "Ganga_all_data":
                taskss = f"results/Ganga_all_data/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)
            
            elif model == "Ganga_org":
                taskss = f"results/Ganga_org/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)
            
            elif model == "Ganga_1024":
                taskss = f"results/Ganga_1024/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)

            elif model.split('-')[0] == 'ganga':
                taskss = f"results/{model}/no_model_name_available/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)
            
            elif model in ["Hinvec", "Hinvec2"]:
                taskss = f"results/{model}/no_model_name_available/no_revision_available/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)
            elif model == "Sailesh97__Hinvec":
                taskss = f"results/{model}/d4fc678720cc1b8c5d18599ce2d9a4d6090c8b6b/{task}.json"

                with open(taskss, 'r') as f:
                    report = json.load(f)

            else:
                taskss = glob.glob(f"{path}{model}/*/{task}.json", recursive=True)
            
                if task == 'MTOPIntentClassification':

                    with open(taskss[1], 'r') as f:

                        report = json.load(f)
                else:

                    with open(taskss[0], 'r') as f:

                        report = json.load(f)

            task_report[task] = report

        
        model_report[model] = task_report

    with open('./results/all_model.json', 'w') as f:

        json.dump(model_report, f, indent=4)


    leader_board = {}

    for model in models:
        leader_board_task = {}
        for task in tasks:

            main_score = []
            if task == "BelebeleRetrieval":
                hf_subsets = ["hin_Deva-hin_Deva", "hin_Deva-eng_Latn", 
                              "eng_Latn-hin_Deva", "hin_Latn-hin_Latn", 
                              "hin_Latn-eng_Latn", "eng_Latn-hin_Latn", 
                              "hin_Deva-hin_Latn", "hin_Latn-hin_Deva"]
                
                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] in hf_subsets:
                        main_score.append(hf['main_score'])
                        
                leader_board_task[task] = np.mean(main_score)
            
            if task == "XQuADRetrieval":

                for hf in model_report[model][task]['scores']['validation']:
                    if hf['hf_subset'] == "hi":
                        leader_board_task[task] = hf['main_score']
                        break
            
            if task == "IndicCrosslingualSTS":

                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "en-hi":
                        leader_board_task[task] = hf['main_score']
                        break

            if task == "WikipediaRerankingMultilingual":

                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "hi":
                        leader_board_task[task] = hf['main_score']
                        break

            if task == "XNLI":
                main_score = []
                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "hi":
                        main_score.append(hf['main_score'])
                        break
                for hf in model_report[model][task]['scores']['validation']:
                    if hf['hf_subset'] == "hi":
                        main_score.append(hf['main_score'])
                        break
                leader_board_task[task] = np.mean(main_score)

            if task == "SIB200ClusteringS2S":

                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "hin_Deva":
                        leader_board_task[task] = hf['main_score']
                        break
            
            if task == "TweetSentimentClassification":

                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "hindi":
                        leader_board_task[task] = hf['main_score']
                        break

            if task == "MultiHateClassification":

                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] == "hin":
                        leader_board_task[task] = hf['main_score']
                        break

            if task == "MTOPIntentClassification":
                main_score = []
                try:
                    for hf in model_report[model][task]['scores']['validation']:
                        if hf['hf_subset'] == "hi":
                            main_score.append(hf['main_score'])
                            
                    for hf in model_report[model][task]['scores']['test']:
                            if hf['hf_subset'] == "hi":
                                main_score.append(hf['main_score'])
                                
                    leader_board_task[task] = np.mean(main_score)
                except KeyError:
                    try:
                        for hf in model_report[model][task]['scores']['test']:
                            if hf['hf_subset'] == "hi":
                                leader_board_task[task] = hf['main_score']
                                break
                    except KeyError:
                        leader_board_task[task] = None

            if task == "IndicLangClassification":

                for hf in model_report[model][task]['scores']['test']:
                    leader_board_task[task] = hf['main_score']
                    break

            if task == "SentimentAnalysisHindi":

                for hf in model_report[model][task]['scores']['train']:
                    leader_board_task[task] = hf['main_score']
                    break

            if task == "HindiDiscourseClassification":

                for hf in model_report[model][task]['scores']['train']:
                    leader_board_task[task] = hf['main_score']
                    break

            if task == "LinceMTBitextMining":

                for hf in model_report[model][task]['scores']['train']:
                    leader_board_task[task] = hf['main_score']
                    break

            if task == "IndicGenBenchFloresBitextMining":
                main_score = []
                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] in ['hin-eng', 'eng-hin']:
                        main_score.append(hf['main_score'])
                    break

                for hf in model_report[model][task]['scores']['validation']:
                    if hf['hf_subset'] in ['hin-eng', 'eng-hin']:
                        main_score.append(hf['main_score'])
                leader_board_task[task] = np.mean(main_score)


            if task == "IN22GenBitextMining":
                main_score = []
                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] in ['eng_Latn-hin_Deva', 'hin_Deva-eng_Latn']:
                        main_score.append(hf['main_score'])
                leader_board_task[task] = np.mean(main_score)

            if task == "IN22ConvBitextMining":
                main_score = []
                for hf in model_report[model][task]['scores']['test']:
                    if hf['hf_subset'] in ['eng_Latn-hin_Deva', 'hin_Deva-eng_Latn']:
                        main_score.append(hf['main_score'])
                leader_board_task[task] = np.mean(main_score)
            

        leader_board[model] = leader_board_task

    with open("./results/leaderboard.json", "w") as f:

        json.dump(leader_board, f, indent=4)

    create_model_scores_csv(leader_board, output_file="model_scores.csv")

            
if __name__ == "__main__":
    main()
    
    
