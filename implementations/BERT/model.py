"""
implementing BERT from scratch
"""

import math
import torch
from transformers.activations import gelu
from transformers import (
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertPreTrainedModel,
    apply_chunking_to_forward,
    set_seed,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)

# print the device using torch.cuda
print(
    f"device:",
    "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
)
torch.set_default_device("cpu")
# set seed for reproducibility
set_seed(42)

# how many labels are we using in training. This is used to decide size of classification head
n_labels = 2

# gelu Activate function
ACT2FN = {"gelu": gelu}

# define BertLayerNorm
BertLayerNorm = torch.nn.LayerNorm

# Let's us create some dummy data for testing the model
# this data will have two classes 0 for negative sentiments and 1 for positive sentiments

input_texts = ["I love cats", "He hates pineapple pizza"]

# Sentiment labels
labels = [1, 0]

# create BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


# create input sequence using the tokenizer
input_sequences = tokenizer(
    text=input_texts,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    retrun_tensors="pt",
)

# now the input sentence is in the form of a dictionary so we can also add labels to it
input_sequences.update({"labels": torch.tensor(labels)})

# since input sequences is a dictionary let's try to print the key value pairs in it
for key, value in input_sequences.items():
    print(f"{key}:{value}")

# print the original input examples
print(f"input_texts:{input_texts}")

# let's see the text also after passing them through the tokenizer
[print(tokenizer.decode(example)) for example in input_sequences["input_ids"]]


# Bert configuration
bert_configuration = BertConfig.from_pretrained("bert-base-cased")

# print the number of layers in the configuration
print(f"Number of layers:{bert_configuration.num_hidden_layers}")

# print the embedding size
print(f"Embedding size:{bert_configuration.hidden_size}")
# see which activation function is used in the config
print(f"Activation function:{bert_configuration.hidden_act}")

# let's create the final model here
model = BertForSequenceClassification.from_pretrained("bert-base-cased")

# perform a forward pass through the model
with torch.no_grad():
    outputs = model(**input_sequences)
print(f"forwward pass outputs:{outputs}")


# creating a class for Bert Embeddings
class BertEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = torch.nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = torch.nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # ADDED
        print("Created Tokens Positions IDs:\n", position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # ADDED
        print("\nTokens IDs:\n", input_ids.shape)
        print("\nTokens Type IDs:\n", token_type_ids.shape)
        print("\nWord Embeddings:\n", inputs_embeds.shape)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)

            # ADDED
            print("\nPosition Embeddings:\n", position_embeddings.shape)

            embeddings += position_embeddings

        # ADDED
        print("\nToken Types Embeddings:\n", token_type_embeddings.shape)
        print("\nSum Up All Embeddings:\n", embeddings.shape)

        embeddings = self.LayerNorm(embeddings)

        # ADDED
        print("\nEmbeddings Layer Nromalization:\n", embeddings.shape)

        embeddings = self.dropout(embeddings)

        # ADDED
        print("\nEmbeddings Dropout Layer:\n", embeddings.shape)

        return embeddings


# Create Bert embedding layer.
bert_embeddings_block = BertEmbeddings(bert_configuration)

# Perform a forward pass.
embedding_output = bert_embeddings_block.forward(
    input_ids=input_sequences["input_ids"],
    token_type_ids=input_sequences["token_type_ids"],
)
