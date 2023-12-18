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


# BErt Encoder
class BertSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # ADDED
        print("Attention Head Size:\n", self.attention_head_size)
        print("\nCombined Attentions Head Size:\n", self.all_head_size)

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # ADDED
        print("\nHidden States:\n", hidden_states.shape)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # ADDED
            print("\nQuery Linear Layer:\n", mixed_query_layer.shape)
            print("\nKey Linear Layer:\n", past_key_value[0].shape)
            print("\nValue Linear Layer:\n", past_key_value[1].shape)

            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            # ADDED
            print("\nQuery Linear Layer:\n", mixed_query_layer.shape)
            print("\nKey Linear Layer:\n", self.key(encoder_hidden_states).shape)
            print("\nValue Linear Layer:\n", self.value(encoder_hidden_states).shape)

            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            # ADDED
            print("\nQuery Linear Layer:\n", mixed_query_layer.shape)
            print("\nKey Linear Layer:\n", self.key(hidden_states).shape)
            print("\nValue Linear Layer:\n", self.value(hidden_states).shape)

            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            # ADDED
            print("\nQuery Linear Layer:\n", mixed_query_layer.shape)
            print("\nKey Linear Layer:\n", self.key(hidden_states).shape)
            print("\nValue Linear Layer:\n", self.value(hidden_states).shape)

            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # ADDED
        print("\nQuery:\n", query_layer.shape)
        print("\nKey:\n", key_layer.shape)
        print("\nValue:\n", value_layer.shape)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # ADDED
        print("\nKey Transposed:\n", key_layer.transpose(-1, -2).shape)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # ADDED
        print("\nAttention Scores:\n", attention_scores.shape)

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd-&amp;amp;amp;gt;bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd-&amp;amp;amp;gt;bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd-&amp;amp;amp;gt;bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # ADDED
        print("\nAttention Scores Divided by Scalar:\n", attention_scores.shape)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # ADDED
        print("\nAttention Probabilities Softmax Layer:\n", attention_probs.shape)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # ADDED
        print("\nAttention Probabilities Dropout Layer:\n", attention_probs.shape)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # ADDED
        print("\nContext:\n", context_layer.shape)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # ADDED
        print("\nContext Permute:\n", context_layer.shape)

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # ADDED
        print("\nContext Reshaped:\n", context_layer.shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Create bert self attention layer.
bert_selfattention_block = BertSelfAttention(bert_configuration)

# Perform a forward pass.
context_embedding = bert_selfattention_block.forward(hidden_states=embedding_output)
