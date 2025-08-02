from transformers import AutoConfig
from transformers.models.mistral import MistralConfig


BIDIR_MISTRAL_TYPE = "bidir_mistral"

class BidirectionalMistralConfig(MistralConfig):
    model_type = BIDIR_MISTRAL_TYPE
    keys_to_ignore_at_inference = ["past_key_values"]

AutoConfig.register(BIDIR_MISTRAL_TYPE, BidirectionalMistralConfig)

BidirectionalMistralConfig.register_for_auto_class()
