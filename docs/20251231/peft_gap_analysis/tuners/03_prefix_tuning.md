# Prefix Tuning

## Overview

Prefix Tuning prepends trainable prefix tokens to the keys and values of each attention layer. Unlike Prompt Tuning which only modifies input embeddings, Prefix Tuning modifies all transformer layers.

## Python Reference

**Files:**
- `peft/src/peft/tuners/prefix_tuning/config.py` (~60 lines)
- `peft/src/peft/tuners/prefix_tuning/model.py` (~200 lines)

### PrefixTuningConfig

```python
@dataclass
class PrefixTuningConfig(PromptLearningConfig):
    encoder_hidden_size: int = None  # Hidden size of prefix encoder
    prefix_projection: bool = False  # Whether to project prefix embeddings
```

**Inherited from PromptLearningConfig:**
- `num_virtual_tokens`: Number of prefix tokens
- `token_dim`: Hidden dimension of base model
- `num_transformer_submodules`: Usually 1 or 2 (encoder, decoder)
- `num_attention_heads`: For splitting prefix
- `num_layers`: Number of transformer layers

### PrefixEncoder

```python
class PrefixEncoder(nn.Module):
    def __init__(self, config):
        self.embedding = nn.Embedding(
            config.num_virtual_tokens,
            config.token_dim
        )

        if config.prefix_projection:
            # Two-layer MLP for reparameterization
            self.transform = nn.Sequential(
                nn.Linear(config.token_dim, config.encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(config.encoder_hidden_size,
                         config.num_layers * 2 * config.token_dim)
            )
        else:
            # Direct embedding for all layers
            self.embedding = nn.Embedding(
                config.num_virtual_tokens,
                config.num_layers * 2 * config.token_dim
            )

    def forward(self, prefix_tokens):
        if self.prefix_projection:
            prefix_embeddings = self.embedding(prefix_tokens)
            past_key_values = self.transform(prefix_embeddings)
        else:
            past_key_values = self.embedding(prefix_tokens)

        # Reshape to (batch, num_layers, 2, num_heads, num_virtual_tokens, head_dim)
        return past_key_values.view(
            batch_size,
            self.num_layers,
            2,  # key and value
            self.num_heads,
            self.num_virtual_tokens,
            self.head_dim
        )
```

### Forward Flow

```python
# In PeftModel forward:
def forward(self, input_ids, **kwargs):
    batch_size = input_ids.shape[0]

    # Get prefix key-value pairs for all layers
    prefix_tokens = torch.arange(self.num_virtual_tokens).expand(batch_size, -1)
    past_key_values = self.prefix_encoder(prefix_tokens)

    # Split into per-layer format
    # Each layer gets (prefix_key, prefix_value)
    past_key_values = split_to_layers(past_key_values)

    # Update attention mask for prefix tokens
    prefix_attention_mask = torch.ones(batch_size, self.num_virtual_tokens)
    attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

    # Forward with past_key_values (prefix is prepended to K,V in each layer)
    return self.base_model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        **kwargs
    )
```

## Elixir Implementation Design

### Module: `HfPeftEx.Tuners.PrefixTuning.Config`

```elixir
defmodule HfPeftEx.Tuners.PrefixTuning.Config do
  @moduledoc """
  Configuration for Prefix Tuning method.
  """

  @derive Jason.Encoder
  defstruct [
    # Base config fields
    peft_type: :prefix_tuning,
    task_type: nil,
    base_model_name_or_path: nil,
    inference_mode: false,
    # Prompt learning config
    num_virtual_tokens: 20,
    token_dim: nil,
    num_transformer_submodules: 1,
    num_attention_heads: nil,
    num_layers: nil,
    # Prefix tuning specific
    encoder_hidden_size: nil,
    prefix_projection: false
  ]

  @type t :: %__MODULE__{
    peft_type: :prefix_tuning,
    num_virtual_tokens: non_neg_integer(),
    token_dim: non_neg_integer() | nil,
    num_layers: non_neg_integer() | nil,
    num_attention_heads: non_neg_integer() | nil,
    encoder_hidden_size: non_neg_integer() | nil,
    prefix_projection: boolean()
  }

  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    struct(__MODULE__, opts)
    |> validate!()
  end

  defp validate!(config) do
    if config.prefix_projection and is_nil(config.encoder_hidden_size) do
      raise ArgumentError, "encoder_hidden_size required when prefix_projection is true"
    end
    config
  end

  @doc """
  Get the total prefix embedding dimension.
  """
  @spec prefix_embedding_dim(t()) :: non_neg_integer()
  def prefix_embedding_dim(config) do
    # num_layers * 2 (key + value) * token_dim
    config.num_layers * 2 * config.token_dim
  end
end
```

### Module: `HfPeftEx.Tuners.PrefixTuning.Encoder`

```elixir
defmodule HfPeftEx.Tuners.PrefixTuning.Encoder do
  @moduledoc """
  Prefix encoder for generating key-value pairs for all layers.
  """

  import Nx.Defn

  defstruct [
    :embedding,          # Base embedding
    :transform,          # Optional MLP projection
    :config,
    :use_projection
  ]

  @type t :: %__MODULE__{}

  @doc """
  Create a new prefix encoder.
  """
  @spec new(config :: struct()) :: t()
  def new(config) do
    if config.prefix_projection do
      new_with_projection(config)
    else
      new_without_projection(config)
    end
  end

  defp new_without_projection(config) do
    # Direct embedding for all layers
    total_dim = config.num_layers * 2 * config.token_dim

    embedding = Nx.random_normal(
      {config.num_virtual_tokens, total_dim},
      0.0, 0.02
    )

    %__MODULE__{
      embedding: embedding,
      transform: nil,
      config: config,
      use_projection: false
    }
  end

  defp new_with_projection(config) do
    # Base embedding
    embedding = Nx.random_normal(
      {config.num_virtual_tokens, config.token_dim},
      0.0, 0.02
    )

    # MLP projection weights
    total_dim = config.num_layers * 2 * config.token_dim
    transform = %{
      w1: Nx.random_normal({config.token_dim, config.encoder_hidden_size}, 0.0, 0.02),
      b1: Nx.broadcast(0.0, {config.encoder_hidden_size}),
      w2: Nx.random_normal({config.encoder_hidden_size, total_dim}, 0.0, 0.02),
      b2: Nx.broadcast(0.0, {total_dim})
    }

    %__MODULE__{
      embedding: embedding,
      transform: transform,
      config: config,
      use_projection: true
    }
  end

  @doc """
  Forward pass to generate past_key_values.
  """
  defn forward(encoder, batch_size) do
    # Get base embeddings
    embeddings = Nx.broadcast(encoder.embedding, {batch_size, encoder.config.num_virtual_tokens, :auto})

    # Apply projection if needed
    past_key_values = if encoder.use_projection do
      apply_projection(embeddings, encoder.transform)
    else
      embeddings
    end

    # Reshape to per-layer format
    reshape_to_layers(past_key_values, encoder.config)
  end

  defnp apply_projection(embeddings, transform) do
    # Two-layer MLP with Tanh
    hidden = Nx.dot(embeddings, transform.w1) |> Nx.add(transform.b1) |> Nx.tanh()
    Nx.dot(hidden, transform.w2) |> Nx.add(transform.b2)
  end

  defnp reshape_to_layers(past_key_values, config) do
    {batch_size, num_tokens, _} = Nx.shape(past_key_values)

    # Reshape to (batch, num_layers, 2, num_heads, num_tokens, head_dim)
    head_dim = div(config.token_dim, config.num_attention_heads)

    past_key_values
    |> Nx.reshape({batch_size, num_tokens, config.num_layers, 2, config.num_attention_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 3, 4, 1, 5])
  end

  @doc """
  Get trainable parameters.
  """
  @spec get_trainable_params(t()) :: map()
  def get_trainable_params(encoder) do
    base = %{"embedding" => encoder.embedding}

    if encoder.use_projection do
      Map.merge(base, %{
        "transform.w1" => encoder.transform.w1,
        "transform.b1" => encoder.transform.b1,
        "transform.w2" => encoder.transform.w2,
        "transform.b2" => encoder.transform.b2
      })
    else
      base
    end
  end
end
```

### Module: `HfPeftEx.Tuners.PrefixTuning.Model`

```elixir
defmodule HfPeftEx.Tuners.PrefixTuning.Model do
  @moduledoc """
  Prefix Tuning model wrapper.
  """

  import Nx.Defn

  defstruct [
    :base_model,
    :config,
    :prefix_encoder
  ]

  @doc """
  Wrap a base model with prefix tuning.
  """
  @spec new(base_model :: map(), config :: struct()) :: t()
  def new(base_model, config) do
    # Auto-fill config from model if needed
    config = maybe_fill_from_model(config, base_model)

    # Create prefix encoder
    prefix_encoder = HfPeftEx.Tuners.PrefixTuning.Encoder.new(config)

    %__MODULE__{
      base_model: base_model,
      config: config,
      prefix_encoder: prefix_encoder
    }
  end

  @doc """
  Forward pass with prefix key-values.
  """
  defn forward(model, input_ids, opts \\ []) do
    batch_size = elem(Nx.shape(input_ids), 0)

    # Generate past_key_values for all layers
    past_key_values = HfPeftEx.Tuners.PrefixTuning.Encoder.forward(
      model.prefix_encoder,
      batch_size
    )

    # Update attention mask
    attention_mask = case opts[:attention_mask] do
      nil -> nil
      mask ->
        prefix_mask = Nx.broadcast(1, {batch_size, model.config.num_virtual_tokens})
        Nx.concatenate([prefix_mask, mask], axis: 1)
    end

    # Forward through base model with past_key_values
    forward_with_past_key_values(model.base_model, input_ids, past_key_values, attention_mask)
  end

  defp maybe_fill_from_model(config, base_model) do
    # Extract model dimensions if not provided
    config
    |> maybe_set(:num_layers, fn -> get_num_layers(base_model) end)
    |> maybe_set(:num_attention_heads, fn -> get_num_heads(base_model) end)
    |> maybe_set(:token_dim, fn -> get_hidden_size(base_model) end)
  end

  defp maybe_set(config, key, value_fn) do
    if Map.get(config, key) do
      config
    else
      Map.put(config, key, value_fn.())
    end
  end

  defp forward_with_past_key_values(base_model, input_ids, past_key_values, attention_mask) do
    # Model-specific: forward with past_key_values
    # Each layer's attention will prepend these to its K,V
    base_model
  end

  defp get_num_layers(model), do: model.config[:num_hidden_layers] || 12
  defp get_num_heads(model), do: model.config[:num_attention_heads] || 12
  defp get_hidden_size(model), do: model.config[:hidden_size] || 768
end
```

## Files to Read

**Python (required reading):**
- `peft/src/peft/tuners/prefix_tuning/config.py`
- `peft/src/peft/tuners/prefix_tuning/model.py`
- `peft/src/peft/config.py` (PromptLearningConfig base)

**Elixir (context):**
- `lib/hf_peft_ex/config.ex`
- `lib/hf_peft_ex/peft_model.ex`

## Tests Required

1. **Config Tests:**
   - Create config with defaults
   - Validate projection requires hidden size
   - Calculate prefix embedding dimension
   - JSON serialization/deserialization

2. **Encoder Tests:**
   - Create without projection (direct embedding)
   - Create with projection (MLP)
   - Forward returns correct shape
   - Trainable params extraction

3. **Model Tests:**
   - Auto-fill config from base model
   - Forward generates past_key_values
   - Attention mask updated correctly
   - Past key-values have correct shape per layer

4. **Integration Tests:**
   - Works with Axon transformer model
   - Prefix prepended to attention correctly

## Mathematical Foundation

**Prefix Tuning Forward:**
```
For each attention layer l:
  K_l = [P_k^l, K_input]  # Prepend prefix keys
  V_l = [P_v^l, V_input]  # Prepend prefix values
  Attention_l = softmax(Q @ K_l^T / sqrt(d)) @ V_l
```

**Where:**
- `P_k^l, P_v^l` are learnable prefix key-value pairs for layer l
- `K_input, V_input` are computed from input
- Prefix is prepended before attention computation

**With Projection (Reparameterization):**
```
P = MLP(E)  # E is base embedding
P = reshape(P, (num_layers, 2, num_heads, num_tokens, head_dim))
```

## Parameter Count

Without projection:
```
params = num_virtual_tokens * num_layers * 2 * token_dim
```

With projection:
```
params = num_virtual_tokens * token_dim +  # base embedding
         token_dim * encoder_hidden_size +  # first linear
         encoder_hidden_size * (num_layers * 2 * token_dim)  # second linear
```

Example (LLaMA-7B, n=20):
- Without: 20 * 32 * 2 * 4096 = 5,242,880 params
- With (hidden=512): Much fewer due to bottleneck

## Key Differences from Prompt Tuning

| Aspect | Prompt Tuning | Prefix Tuning |
|--------|---------------|---------------|
| Where applied | Input embeddings | All attention layers |
| Affects | Initial representation | All layer representations |
| Parameter count | n * d | n * L * 2 * d |
| Complexity | Lower | Higher |
| Performance | Good for simple tasks | Better for complex tasks |

## Complexity

- **Implementation:** Medium-High
- **Mathematical:** Medium
- **Dependencies:** Model architecture knowledge (num_layers, heads)

## Priority

**High** - Demonstrates per-layer prefix injection pattern
