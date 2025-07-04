import os
import json
import random
import glob

import numpy as np
import pandas as pd

from datasets import load_dataset, load_from_disk

INSTRUCTION = {
    "bengali":	("নির্দেশাবলী: প্রদত্ত পাঠ্যের জন্য সবচেয়ে প্রাসঙ্গিক সারাংশ খুঁজুন। পাঠ্য:", "নির্দেশাবলী: প্রদত্ত পাঠ্যের জন্য সবচেয়ে প্রাসঙ্গিক নথি খুঁজুন। পাঠ্য:"),
    "burmese":	("လမ်းညွှန်ချက်များ- ပေးထားသော စာသားအတွက် အကိုက်ညီဆုံး အနှစ်ချုပ်ကို ရှာပါ။ စာသား-", "လမ်းညွှန်ချက်များ- ပေးထားသော စာသားအတွက် အသက်ဆိုင်ဆုံး စာရွက်စာတမ်းကို ရှာပါ။ စာသား-"),
    "english":	("Instruction: Find the most relevant summary for the given text. Text:", "Instruction: Find the most relevant document for the given text. Text:"),
    "gujarati": ("દિશાઓ: આપેલ ટેક્સ્ટ માટે સૌથી સુસંગત સારાંશ શોધો. ટેક્સ્ટ:", "દિશા-નિર્દેશો: આપેલ ટેક્સ્ટ માટે સૌથી સુસંગત દસ્તાવેજ શોધો. ટેક્સ્ટ:"),
    "hindi": ("निर्देश: दिए गए पाठ के लिए सबसे प्रासंगिक सारांश प्राप्त करें। पाठ:", "निर्देश: दिए गए पाठ के लिए सबसे प्रासंगिक दस्तावेज़ खोजें। पाठ:"),
    "marathi":	("दिशानिर्देश: दिलेल्या मजकुरासाठी सर्वात संबंधित सारांश शोधा. मजकूर:", "दिशानिर्देश: दिलेल्या मजकुरासाठी सर्वात संबंधित दस्तऐवज शोधा. मजकूर:"),
    "nepali": ("निर्देशनहरू: दिइएको पाठको लागि सबैभन्दा सान्दर्भिक सारांश खोज्नुहोस्। पाठ:", "निर्देशनहरू: दिइएको पाठको लागि सबैभन्दा सान्दर्भिक कागजात फेला पार्नुहोस्। पाठ:"),
    "punjabi": ("ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼: ਦਿੱਤੇ ਟੈਕਸਟ ਲਈ ਸਭ ਤੋਂ ਢੁਕਵਾਂ ਸੰਖੇਪ ਲੱਭੋ। ਟੈਕਸਟ:", "ਦਿਸ਼ਾ-ਨਿਰਦੇਸ਼: ਦਿੱਤੇ ਗਏ ਟੈਕਸਟ ਲਈ ਸਭ ਤੋਂ ਢੁਕਵਾਂ ਦਸਤਾਵੇਜ਼ ਲੱਭੋ। ਟੈਕਸਟ:"),
    "tamil": ("திசைகள்: கொடுக்கப்பட்ட உரைக்கு மிகவும் பொருத்தமான சுருக்கத்தைக் கண்டறியவும். உரை:", "திசைகள்: கொடுக்கப்பட்ட உரைக்கு மிகவும் பொருத்தமான ஆவணத்தைக் கண்டறியவும். உரை:"),
    "telugu": ("దిశలు: అందించిన వచనానికి అత్యంత సంబంధిత సారాంశాన్ని కనుగొనండి. వచనం:", "దిశలు: అందించిన వచనానికి అత్యంత సంబంధిత పత్రాన్ని కనుగొనండి. వచనం:"),
    "urdu":	("ہدایات: دیے گئے متن کے لیے سب سے زیادہ متعلقہ خلاصہ تلاش کریں۔ متن:", "ہدایات: دیے گئے متن کے لیے سب سے زیادہ متعلقہ دستاویز تلاش کریں۔ متن:"),
    }

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

if __name__ == "__main__":

    files = glob.glob("./Bitext Mining/CrossSum/*")
    for file in files:
        [slang, tlang] = file.split("/")[-1].split('-')
        print(slang, tlang)

        train_data = load_jsonl(f"{file}/{slang}-{tlang}_train.jsonl")
        val_data = load_jsonl(f"{file}/{slang}-{tlang}_val.jsonl")

        for dtype, data in enumerate([train_data, val_data]):
            id_num = 0
            data_json = []
            for idx, sample in enumerate(data):

                if idx%2==0:
                    sample = {
                        "id": f"crosssum_{slang}_{tlang}_{id_num}",
                        "source": f"{INSTRUCTION[slang][0]} {sample['text']}",
                        "target": sample["summary"],
                    }
                else:
                    sample = {
                        "id": f"crosssum_{slang}_{tlang}_{id_num}",
                        "source": f"{INSTRUCTION[tlang][1]} {sample['summary']}",
                        "target": sample["summary"],
                    }
                id_num+=1

                data_json.append(sample)
            if dtype == 0:
                save_jsonl(data_json, f"../Processed_data/crosssum_{slang}_{tlang}.jsonl")
            else:
                save_jsonl(data_json, f"../Processed_data/crosssum_{slang}_{tlang}_test.jsonl")