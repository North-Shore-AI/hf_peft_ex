# Prompt Tuning

## Overview

Prompt Tuning adds learnable virtual tokens (soft prompts) to the input embedding. These virtual tokens are prepended to the input and learned during training while the base model remains frozen.

## Python Reference

**Files:**
- `peft/src/peft/tuners/prompt_tuning/config.py` (~100 lines)
- `peft/src/peft/tuners/prompt_tuning/model.py` (~150 lines)

### PromptTuningConfig

```python
@dataclass
class PromptTuningConfig(PromptLearningConfig):
    prompt_tuning_init: Union[PromptTuningInit, str] = PromptTuningInit.RANDOM
    prompt_tuning_init_text: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

class PromptTuningInit(str, Enum):
    TEXT = "TEXT"           # Initialize from text embeddings
    SAMPLE_VOCAB = "SAMPLE_VOCAB"  # Sample from vocabulary
    RANDOM = "RANDOM"       # Random continuous vectors
```

**Inherited from PromptLearningConfig:**
- `num_virtual_tokens`: Number of soft prompt tokens
- `token_dim`: Hidden embedding dimension
- `num_transformer_submodules`: Number of transformer layers
- `num_attention_heads`: Attention heads in base model
- `num_layers`: Number of transformer layers

### PromptEmbedding

```python
class PromptEmbedding(nn.Module):
    def __init__(self, config, word_embeddings):
        self.embedding = nn.Embedding(
            config.num_virtual_tokens,
            config.token_dim
        )

        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            # Initialize from tokenized text
            tokens = tokenizer(config.prompt_tuning_init_text)
            word_embedding_weights = word_embeddings(tokens)
            self.embedding.weight = nn.Parameter(word_embedding_weights)

        elif config.prompt_tuning_init == PromptTuningInit.SAMPLE_VOCAB:
            # Sample random vocabulary tokens
            indices = torch.randint(0, word_embeddings.num_embeddings,
                                   (config.num_virtual_tokens,))
            word_embedding_weights = word_embeddings(indices)
            self.embedding.weight = nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Return soft prompt embeddings
        return self.embedding(indices)
```

### Forward Flow

```python
# In PeftModel forward:
def forward(self, input_ids, **kwargs):
    batch_size = input_ids.shape[0]

    # Get base embeddings
    inputs_embeds = self.word_embeddings(input_ids)

    # Get soft prompt embeddings
    prompt_tokens = torch.arange(self.num_virtual_tokens).to(device)
    prompt_tokens = prompt_tokens.expand(batch_size, -1)
    prompt_embeds = self.prompt_encoder(prompt_tokens)

    # Concatenate: [soft_prompts, input_embeddings]
    inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

    # Update attention mask
    prefix_attention_mask = torch.ones(batch_size, self.num_virtual_tokens)
    attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

    # Forward through base model with modified embeddings
    return self.base_model(inputs_embeds=inputs_embeds,
                          attention_mask=attention_mask, **kwargs)
```

## Elixir Implementation Design

### Module: `HfPeftEx.Tuners.PromptTuning.Config`

```elixir
defmodule HfPeftEx.Tuners.PromptTuning.Config do
  @moduledoc """
  Configuration for Prompt Tuning method.
  """

  @derive Jason.Encoder
  defstruct [
    # Base config fields
    peft_type: :prompt_tuning,
    task_type: nil,
    base_model_name_or_path: nil,
    inference_mode: false,
    # Prompt learning config
    num_virtual_tokens: 20,
    token_dim: nil,           # Set from model config
    num_transformer_submodules: nil,
    num_attention_heads: nil,
    num_layers: nil,
    # Prompt tuning specific
    prompt_tuning_init: :random,  # :random | :text | :sample_vocab
    prompt_tuning_init_text: nil,
    tokenizer_name_or_path: nil,
    tokenizer_kwargs: nil
  ]

  @type init_method :: :random | :text | :sample_vocab

  @type t :: %__MODULE__{
    peft_type: :prompt_tuning,
    num_virtual_tokens: non_neg_integer(),
    prompt_tuning_init: init_method(),
    prompt_tuning_init_text: String.t() | nil,
    tokenizer_name_or_path: String.t() | nil
  }

  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct(__MODULE__, opts)
    |> validate!()
  end

  defp validate!(config) do
    if config.prompt_tuning_init == :text and is_nil(config.prompt_tuning_init_text) do
      raise ArgumentError, "prompt_tuning_init_text required when prompt_tuning_init is :text"
    end

    if config.prompt_tuning_init == :text and is_nil(config.tokenizer_name_or_path) do
      raise ArgumentError, "tokenizer_name_or_path required when prompt_tuning_init is :text"
    end

    if config.num_virtual_tokens < 1 do
      raise ArgumentError, "num_virtual_tokens must be at least 1"
    end

    config
  end
end
```

### Module: `HfPeftEx.Tuners.PromptTuning.Embedding`

```elixir
defmodule HfPeftEx.Tuners.PromptTuning.Embedding do
  @moduledoc """
  Learnable soft prompt embedding for Prompt Tuning.
  """

  import Nx.Defn

  defstruct [
    :embedding,           # Nx.Tensor of shape {num_virtual_tokens, token_dim}
    :num_virtual_tokens,
    :token_dim
  ]

  @type t :: %__MODULE__{
    embedding: Nx.Tensor.t(),
    num_virtual_tokens: non_neg_integer(),
    token_dim: non_neg_integer()
  }

  @doc """
  Create a new prompt embedding with specified initialization.
  """
  @spec new(config :: struct(), word_embeddings :: Nx.Tensor.t() | nil) :: t()
  def new(config, word_embeddings \\ nil) do
    {num_virtual_tokens, token_dim} = {config.num_virtual_tokens, config.token_dim}

    embedding = case config.prompt_tuning_init do
      :random ->
        # Random initialization with small values
        Nx.random_normal({num_virtual_tokens, token_dim}, 0.0, 0.02)

      :sample_vocab when not is_nil(word_embeddings) ->
        # Sample random tokens from vocabulary
        vocab_size = elem(Nx.shape(word_embeddings), 0)
        indices = Nx.random_uniform({num_virtual_tokens}, 0, vocab_size, type: :s64)
        Nx.take(word_embeddings, indices, axis: 0)

      :text when not is_nil(word_embeddings) ->
        # Initialize from tokenized text
        # Note: Requires tokenizer integration
        raise "Text initialization requires tokenizer - use from_text/3 instead"

      _ ->
        Nx.random_normal({num_virtual_tokens, token_dim}, 0.0, 0.02)
    end

    %__MODULE__{
      embedding: embedding,
      num_virtual_tokens: num_virtual_tokens,
      token_dim: token_dim
    }
  end

  @doc """
  Create prompt embedding initialized from text.
  """
  @spec from_text(config :: struct(), text :: String.t(), tokenize_fn :: function()) :: t()
  def from_text(config, text, tokenize_fn) when is_function(tokenize_fn, 1) do
    # Get embeddings from tokenized text
    embeddings = tokenize_fn.(text)

    # Ensure we have exactly num_virtual_tokens
    {actual, token_dim} = Nx.shape(embeddings)
    embedding = cond do
      actual == config.num_virtual_tokens ->
        embeddings
      actual > config.num_virtual_tokens ->
        Nx.slice(embeddings, [0, 0], [config.num_virtual_tokens, token_dim])
      actual < config.num_virtual_tokens ->
        # Pad with random or repeat
        padding = Nx.random_normal({config.num_virtual_tokens - actual, token_dim}, 0.0, 0.02)
        Nx.concatenate([embeddings, padding], axis: 0)
    end

    %__MODULE__{
      embedding: embedding,
      num_virtual_tokens: config.num_virtual_tokens,
      token_dim: token_dim
    }
  end

  @doc """
  Get prompt embeddings for a batch.
  """
  defn forward(prompt_embedding, batch_size) do
    # Expand to batch: {num_virtual_tokens, token_dim} -> {batch_size, num_virtual_tokens, token_dim}
    Nx.broadcast(prompt_embedding.embedding, {batch_size, prompt_embedding.num_virtual_tokens, prompt_embedding.token_dim})
  end

  @doc """
  Get trainable parameters.
  """
  @spec get_trainable_params(t()) :: %{String.t() => Nx.Tensor.t()}
  def get_trainable_params(prompt_embedding) do
    %{"embedding" => prompt_embedding.embedding}
  end
end
```

### Module: `HfPeftEx.Tuners.PromptTuning.Model`

```elixir
defmodule HfPeftEx.Tuners.PromptTuning.Model do
  @moduledoc """
  Prompt Tuning model wrapper.
  """

  import Nx.Defn

  defstruct [
    :base_model,
    :config,
    :prompt_encoder,
    :word_embeddings
  ]

  @doc """
  Wrap a base model with prompt tuning.
  """
  @spec new(base_model :: map(), config :: struct()) :: t()
  def new(base_model, config) do
    # Extract word embeddings from base model
    word_embeddings = get_word_embeddings(base_model)

    # Create prompt encoder
    prompt_encoder = HfPeftEx.Tuners.PromptTuning.Embedding.new(
      config,
      word_embeddings
    )

    %__MODULE__{
      base_model: base_model,
      config: config,
      prompt_encoder: prompt_encoder,
      word_embeddings: word_embeddings
    }
  end

  @doc """
  Forward pass with soft prompts prepended.
  """
  defn forward(model, input_ids, opts \\ []) do
    batch_size = elem(Nx.shape(input_ids), 0)

    # Get input embeddings
    inputs_embeds = get_input_embeddings(model.base_model, input_ids)

    # Get soft prompt embeddings
    prompt_embeds = HfPeftEx.Tuners.PromptTuning.Embedding.forward(
      model.prompt_encoder,
      batch_size
    )

    # Concatenate: [prompts, inputs]
    combined_embeds = Nx.concatenate([prompt_embeds, inputs_embeds], axis: 1)

    # Update attention mask if provided
    attention_mask = case opts[:attention_mask] do
      nil -> nil
      mask ->
        prefix_mask = Nx.broadcast(1, {batch_size, model.config.num_virtual_tokens})
        Nx.concatenate([prefix_mask, mask], axis: 1)
    end

    # Forward through base model with inputs_embeds
    forward_base_model(model.base_model, combined_embeds, attention_mask)
  end

  defp get_word_embeddings(base_model) do
    # Model-specific: extract embedding layer
    base_model[:embeddings] || base_model[:embed_tokens]
  end

  defp get_input_embeddings(base_model, input_ids) do
    # Model-specific: apply embedding lookup
    Nx.take(get_word_embeddings(base_model), input_ids, axis: 0)
  end

  defp forward_base_model(base_model, inputs_embeds, attention_mask) do
    # Model-specific: forward with inputs_embeds instead of input_ids
    # This needs integration with actual Axon models
    base_model
  end
end
```

## Files to Read

**Python (required reading):**
- `peft/src/peft/tuners/prompt_tuning/config.py`
- `peft/src/peft/tuners/prompt_tuning/model.py`
- `peft/src/peft/config.py` (PromptLearningConfig base)

**Elixir (context):**
- `lib/hf_peft_ex/config.ex`
- `lib/hf_peft_ex/peft_model.ex`

## Tests Required

1. **Config Tests:**
   - Create config with defaults
   - Validate text init requires tokenizer
   - JSON serialization/deserialization

2. **Embedding Tests:**
   - Random initialization creates correct shape
   - Sample vocab initialization works
   - Text initialization (mocked tokenizer)
   - Forward returns correct batch shape

3. **Model Tests:**
   - Wrap base model successfully
   - Forward prepends prompts correctly
   - Attention mask updated correctly
   - Output shape accounts for virtual tokens

4. **Training Tests:**
   - Prompt embeddings are trainable
   - Base model params are frozen

## Mathematical Foundation

**Prompt Tuning Forward:**
```
X_input = [P_1, P_2, ..., P_n, E_1, E_2, ..., E_m]
```

**Where:**
- `P_i` are learnable soft prompt embeddings
- `E_i` are input token embeddings
- `n` is num_virtual_tokens
- `m` is input sequence length

**Total sequence length:** `n + m`

## Parameter Count

For `n` virtual tokens with dimension `d`:
- Parameters = `n * d`
- Typically `n = 20`, `d = 768` (BERT) or `d = 4096` (LLaMA)
- Example: 20 * 4096 = 81,920 parameters

## Key Differences from LoRA

| Aspect | Prompt Tuning | LoRA |
|--------|---------------|------|
| Where applied | Input embeddings only | All target layers |
| Parameter type | Continuous vectors | Low-rank matrices |
| Sequence length | Extended by n tokens | Unchanged |
| Computational cost | Higher attention (longer seq) | Higher per-layer |

## Complexity

- **Implementation:** Medium
- **Mathematical:** Low
- **Dependencies:** Base model embedding access

## Priority

**High** - Different paradigm from LoRA, validates prompt learning support
