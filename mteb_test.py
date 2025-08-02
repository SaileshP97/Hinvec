import torch
import argparse

from transformers import AutoModel, AutoTokenizer, AutoConfig
import mteb
from mteb.encoder_interface import PromptType
import numpy as np
from sentence_transformers import SentenceTransformer
import huggingface_hub
import os

from ganga_modeling import EmbeddingModel, BidirectionalMistralModel, BidirectionalMistralConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CustomModel:

    def __init__(self, base_model, tokenizer, pooling_type="mean"):
        super().__init__()
        self.model = base_model.to("cuda")
        self.pooling_type = pooling_type
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.tokenizer.padding_side = "right"

    def __get_task_prompt__(self, task_name):

        hindi_prompts = {
            "Any2AnyMultilingualRetrieval": "कृपया दिए गए टेक्स्ट के लिए संबंधित जानकारी पुनः प्राप्त करें।",
            
            "XM3600T2IRetrieval": "दिए गए टेक्स्ट के आधार पर उपयुक्त छवि खोजें।",

            "BibleNLPBitextMining": "बाइबल के अनुवादित वाक्यों को खोजें और उनका मिलान करें।",
            "FloresBitextMining": "दी गई भाषा के समानार्थी वाक्य जोड़ियों को खोजें।",
            "IN22ConvBitextMining": "निर्देश: दिए गए वाक्य के लिए अनुवादित वाक्य जोड़े खोजें। वाक्य: ",
            "IN22GenBitextMining": "निर्देश: दिए गए पाठ का सबसे समान अनुवाद खोजें। पाठ: ",
            "IndicGenBenchFloresBitextMining": "निर्देश: एक प्रश्न दिया गया है, सबसे समान अनुच्छेद को पुनः प्राप्त करें। प्रश्न: ",
            "LinceMTBitextMining":  "Instruction: Find the most similar romanised Hindi sentence of the give english sentence. Sentence: ",
            "NTREXBitextMining": "NTREX डेटासेट में समानार्थक वाक्य खोजें।",
            "PhincBitextMining": "Phinc डेटासेट से द्विभाषी टेक्स्ट जोड़ियों को खोजें।",
            "Tatoeba": "Tatoeba डेटासेट में अनुवादित वाक्य जोड़ियों को खोजें।",
            "WebFAQBitextMiningQuestions": "WebFAQ डेटासेट से समान प्रश्न जोड़ियों को खोजें।",
            "WebFAQBitextMiningQAs": "WebFAQ डेटासेट से समान प्रश्न-उत्तर जोड़ियों को खोजें।",

            "HindiDiscourseClassification": "निर्देश: दिए गए हिंदी पाठ को निम्नलिखित में से किसी एक श्रेणी में वर्गीकृत करें: 'वर्णनात्मक', 'कथात्मक', 'संवाद', 'तर्कपूर्ण', 'सूचनात्मक', या 'अन्य'। पाठ: ",
            "SentimentAnalysisHindi": "निर्देश: दिए गए पाठ को निम्नलिखित भावना श्रेणियों में से किसी एक में वर्गीकृत करें: सकारात्मक, नकारात्मक, या तटस्थ। पाठ: ",
            "IndicLangClassification": "निर्देश: दिए गए पाठ की भाषा को निम्नलिखित भाषाओं में से किसी एक के रूप में वर्गीकृत करें: असमिया (as), बांग्ला (bn), गुजराती (gu), हिंदी (hi), कन्नड़ (kn), मलयालम (ml), मराठी (mr), उड़िया (or), पंजाबी (pa), तमिल (ta), या तेलुगू (te)। पाठ: ",
            "IndicSentimentClassification": "दिए गए टेक्स्ट की भावना का विश्लेषण करें।",
            "LanguageClassification": "दिए गए टेक्स्ट की भाषा निर्धारित करें।",
            "MassiveIntentClassification": "दिए गए टेक्स्ट का उद्देश्य निर्धारित करें।",
            "MassiveScenarioClassification": "दिए गए टेक्स्ट के लिए परिदृश्य श्रेणी निर्धारित करें।",
            "MTOPDomainClassification": "निर्देश: दिए गए पाठ को उसके डोमेन के आधार पर वर्गीकृत करें। पाठ: ",
            "MTOPIntentClassification": "निर्देश: दिए गए पाठ को उसके उद्देश्य के आधार पर वर्गीकृत करें। पाठ: ",
            "MultiHateClassification": "निर्देश: दिए गए पाठ को घृणास्पद भाषण वाले के रूप में वर्गीकृत करें या नहीं। पाठ:",
            "SIB200Classification": "दिए गए टेक्स्ट को SIB200 श्रेणी में वर्गीकृत करें।",
            "TweetSentimentClassification": "निर्देश: दिए गए पाठ को निम्नलिखित भावना श्रेणियों में से किसी एक में वर्गीकृत करें: नकारात्मक, तटस्थ या सकारात्मक। पाठ:",

            "IndicReviewsClusteringP2P": "दिए गए समीक्षाओं को समान अर्थ वाले समूहों में व्यवस्थित करें।",
            "SIB200ClusteringS2S": "निर्देश: दिए गए वाक्यों के आधार पर सबसे समान वाक्य ढूंढें: वाक्य: ",

            "PubChemWikiPairClassification": "दिए गए दो वाक्यों के बीच संबंध को वर्गीकृत करें।",
            "XNLI": "Instruction: Given a query, retrieve the most similar sentence. Query: ",

            "MIRACLReranking": "दिए गए क्वेरी और दस्तावेज़ की प्रासंगिकता के आधार पर दस्तावेज़ों को पुनः क्रमित करें।",
            "WikipediaRerankingMultilingual": "निर्देश: दिए गए प्रश्न के लिए सबसे अधिक समानता रखने वाले दस्तावेज़ को पहचानें और पुनः प्राप्त करें। प्रश्न:",

            "BelebeleRetrieval": "निर्देश: एक प्रश्न दिया गया है, सबसे समान अनुच्छेद को पुनः प्राप्त करें। प्रश्न: ",
            "IndicQARetrieval": "दिए गए प्रश्न के लिए उपयुक्त उत्तर खोजें।",
            "MintakaRetrieval": "Mintaka डेटासेट से प्रासंगिक उत्तर खोजें।",
            "MIRACLRetrieval": "MIRACL डेटासेट से उपयुक्त दस्तावेज़ पुनः प्राप्त करें।",
            "MIRACLRetrievalHardNegatives": "MIRACL डेटासेट से कठिन नकारात्मक उदाहरणों के साथ पुनः प्राप्ति करें।",
            "MLQARetrieval": "निर्देश: सबसे प्रासंगिक उत्तर खोजने के लिए क्वेरी दें: क्वेरी:",
            "MultiLongDocRetrieval": "बहु-भाषी लंबी दस्तावेज़ पुनः प्राप्ति करें।",
            "WebFAQRetrieval": "WebFAQ डेटासेट से प्रासंगिक उत्तर खोजें।",
            "WikipediaRetrievalMultilingual": "निर्देश: दिए गए प्रश्न में से सबसे प्रासंगिक अनुच्छेद को पुनः प्राप्त करें। प्रश्न:",
            "XPQARetrieval": "निर्देश: सबसे प्रासंगिक उत्तर खोजने के लिए क्वेरी दें: क्वेरी:",
            "XQuADRetrieval": "निर्देश: सबसे प्रासंगिक उत्तर खोजने के लिए क्वेरी दें: क्वेरी:",

            "IndicCrosslingualSTS": "निर्देश: दिए गए वाक्य युग्मों के बीच अर्थगत समानता का मूल्यांकन करें। पाठ: ",
            "SemRel24STS": "दिए गए वाक्य युग्मों के बीच अर्थ समानता की गणना करें।"
        }

        return hindi_prompts[task_name]
        
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        if prompt_type == 'query':
            prompt = self.__get_task_prompt__(task_name)
            input_text = [f"Instruction: {prompt}\n Query: {sent}" for sent in sentences]
        else:
            input_text = sentences

        embeddings_list = []
        batch_size = 128
        for i in range(0, len(input_text), batch_size):
            batch = input_text[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(**inputs)
                
            if self.pooling_type == "mean":

                token_embeddings = outputs.last_hidden_state
                if self.model.config.model_type == "x":
                    attention_mask = torch.ones(100).to(inputs['attention_mask'].device)
                else:
                    attention_mask = inputs['attention_mask']

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

            elif self.pooling_type == "cls":
                batch_embeddings = outputs.last_hidden_state[:, 0]

            elif self.pooling_type == "eos":
                batch_embeddings = outputs.last_hidden_state[:, -1]

            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings_list.append(batch_embeddings.cpu().numpy())
        
        all_embeddings = np.vstack(embeddings_list)
        return all_embeddings

def main():

    parser = argparse.ArgumentParser(description="Testing Embedding model on MTEB benchmark.")

    parser.add_argument("--model_name", type=str, required=True, help="Model to be evaluated.")
    parser.add_argument("--pooling_type", type=str, required=True, default="mean", help="Pooling Type.")
    parser.add_argument("--bi_dir", action="store_true", help="Whether to use bidirectional model or not.")

    args = parser.parse_args()

    if args.bi_dir:
        state_dict = torch.load(f"{args.model_name}/pytorch_model.bin")

        base_model = AutoModel.from_pretrained("LingoIITGN/Ganga-2-1B")
        original_config = AutoConfig.from_pretrained("LingoIITGN/Ganga-2-1B")
        bidir_config = BidirectionalMistralConfig(**original_config.to_dict())
        base_model = BidirectionalMistralModel(bidir_config)
        base_model.load_state_dict(state_dict)

    else:
        base_model = AutoModel.from_pretrained(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained("LingoIITGN/Ganga-2-1B")

    model = CustomModel(base_model, tokenizer, pooling_type=args.pooling_type)

    tasks = mteb.get_tasks(tasks=["IN22ConvBitextMining",
    "IN22GenBitextMining",
    "IndicGenBenchFloresBitextMining",
    "LinceMTBitextMining",
    "SIB200ClusteringS2S",
    "HindiDiscourseClassification",
    "SentimentAnalysisHindi",
    "IndicLangClassification",
    "MTOPIntentClassification",
    "MultiHateClassification",
    "TweetSentimentClassification",
    "XNLI",
    "BelebeleRetrieval",
    "XQuADRetrieval",
    "WikipediaRerankingMultilingual",
    "IndicCrosslingualSTS"], languages=['hin'])

    #tasks = mteb.get_tasks(tasks=["MIRACLRetrieval"], languages=['hin'])
    
    model_name = args.model_name.split('/')[1]
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, encode_kwargs={"batch_size": 3000}, output_folder=f"results/{model_name}")

if __name__ == "__main__":
    main()