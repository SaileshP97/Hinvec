import os
import json
import random
import glob

import numpy as np
import pandas as pd

from datasets import load_dataset, load_from_disk

INSTRUCTION = {
    "as": "নিৰ্দেশনা: প্ৰদত্ত পাঠটোৰ বাবে আটাইতকৈ মিল থকা অনুবাদটো বিচাৰক। পাঠ্য:",
    "bn": "নির্দেশনা: প্রদত্ত লেখাটির জন্য সবচেয়ে অনুরূপ অনুবাদটি খুঁজুন। লেখা:",
    "en": "Instruction: Find the most similar translation for the given text. Text:",
    "gu": "સૂચના: આપેલ ટેક્સ્ટ માટે સૌથી સમાન અનુવાદ શોધો. ટેક્સ્ટ:",
    "hi": "निर्देश: दिए गए पाठ के लिए सबसे समान अनुवाद खोजें। पाठ:",
    "kn": "ಸೂಚನೆ: ಕೊಟ್ಟಿರುವ ಪಠ್ಯಕ್ಕೆ ಹೆಚ್ಚು ಹೋಲುವ ಅನುವಾದವನ್ನು ಹುಡುಕಿ. ಪಠ್ಯ:",
    "ml": "നിർദ്ദേശം: നൽകിയിരിക്കുന്ന വാചകത്തിന് ഏറ്റവും സമാനമായ വിവർത്തനം കണ്ടെത്തുക. വാചകം:",
    "mr": "सूचना: दिलेल्या मजकुरासाठी सर्वात समान भाषांतर शोधा. मजकूर:",
    "or": "ନିର୍ଦ୍ଦେଶ: ଦିଆଯାଇଥିବା ପାଠ୍ୟ ପାଇଁ ସବୁଠାରୁ ସମାନ ଅନୁବାଦ ଖୋଜନ୍ତୁ। ପାଠ୍ୟ:",
    "pa": "ਹਦਾਇਤ: ਦਿੱਤੇ ਗਏ ਟੈਕਸਟ ਲਈ ਸਭ ਤੋਂ ਮਿਲਦਾ-ਜੁਲਦਾ ਅਨੁਵਾਦ ਲੱਭੋ। ਟੈਕਸਟ:",
    "ta": "வழிமுறை: கொடுக்கப்பட்ட உரைக்கு மிகவும் ஒத்த மொழிபெயர்ப்பைக் கண்டறியவும். உரை:",
    "te": "సూచన: ఇచ్చిన పాఠ్యానికి అత్యంత సారూప్యమైన అనువాదాన్ని కనుగొనండి. పాఠ్యం:",
    }

LANGUAGE_CODE = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

def load_jsonl(path):

    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line)) 
    return data

def save_jsonl(data, path):

    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

def main():

    for lang in LANGUAGE_CODE:
        print(f"Downloading {lang} data")

        ds = load_dataset("ai4bharat/Samanantar", lang)
        data = [sample for sample in ds["train"]]

        os.makedirs("./Translation/Samanantar", exist_ok=True)
        random.shuffle(data)

        save_jsonl(data, f"./Translation/Samanantar/{lang}.jsonl")
        save_jsonl(data, "../Processed_data/samanantar.jsonl")

if __name__ == "__main__":
    main()