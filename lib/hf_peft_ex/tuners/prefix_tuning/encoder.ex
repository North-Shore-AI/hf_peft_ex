defmodule HfPeftEx.Tuners.PrefixTuning.Encoder do
  @moduledoc """
  Prefix Encoder for Prefix Tuning.

  The PrefixEncoder creates trainable key-value prefix embeddings that are
  prepended to each attention layer's key and value matrices.

  ## Architecture

  ### Without Projection (Direct Embedding)
  A single embedding matrix of shape `(num_virtual_tokens, num_layers * 2 * token_dim)`
  that directly stores all prefix key-value pairs for all layers.

  ### With Projection (Reparameterization)
  Uses a two-layer MLP for reparameterization:
  1. Embedding: `(num_virtual_tokens, token_dim)`
  2. Transform: `Linear(token_dim, hidden) -> Tanh -> Linear(hidden, output_dim)`

  Where `output_dim = num_layers * 2 * token_dim`.

  The projection approach typically uses fewer parameters and can improve training
  stability through the bottleneck architecture.

  ## Output Shape

  The forward pass returns past_key_values with shape:
  `(batch_size, num_layers, 2, num_heads, num_virtual_tokens, head_dim)`

  Where dimension 2 contains [key, value] pairs for each layer.
  """

  import Nx.Defn

  alias HfPeftEx.Tuners.PrefixTuning.Config

  @type transform :: %{
          w1: Nx.Tensor.t(),
          b1: Nx.Tensor.t(),
          w2: Nx.Tensor.t(),
          b2: Nx.Tensor.t()
        }

  @type t :: %__MODULE__{
          config: Config.t(),
          embedding: Nx.Tensor.t(),
          use_projection: boolean(),
          transform: transform() | nil
        }

  defstruct [:config, :embedding, :use_projection, :transform]

  @doc """
  Creates a new PrefixEncoder from a configuration.

  The encoder stores trainable embeddings that generate past_key_values
  for all transformer layers.

  ## Examples

      config = Config.new(
        num_virtual_tokens: 20,
        num_layers: 12,
        token_dim: 768,
        num_attention_heads: 12
      )
      encoder = Encoder.new(config)

  """
  @spec new(Config.t()) :: t()
  def new(%Config{} = config) do
    output_dim = config.num_layers * 2 * config.token_dim

    if config.prefix_projection do
      # With projection: small embedding + MLP transform
      embedding = init_embedding(config.num_virtual_tokens, config.token_dim)

      transform = %{
        w1: init_weight(config.token_dim, config.encoder_hidden_size),
        b1: init_bias(config.encoder_hidden_size),
        w2: init_weight(config.encoder_hidden_size, output_dim),
        b2: init_bias(output_dim)
      }

      %__MODULE__{
        config: config,
        embedding: embedding,
        use_projection: true,
        transform: transform
      }
    else
      # Without projection: direct large embedding
      embedding = init_embedding(config.num_virtual_tokens, output_dim)

      %__MODULE__{
        config: config,
        embedding: embedding,
        use_projection: false,
        transform: nil
      }
    end
  end

  defp init_embedding(num_tokens, dim) do
    # Initialize with small random values (normal distribution scaled down)
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.normal(key, 0.0, 0.02, shape: {num_tokens, dim}, type: :f32)
    tensor
  end

  defp init_weight(in_dim, out_dim) do
    # Xavier/Glorot initialization
    key = Nx.Random.key(System.system_time() + :rand.uniform(1000))
    stddev = :math.sqrt(2.0 / (in_dim + out_dim))
    {tensor, _key} = Nx.Random.normal(key, 0.0, stddev, shape: {in_dim, out_dim}, type: :f32)
    tensor
  end

  defp init_bias(dim) do
    Nx.broadcast(0.0, {dim}) |> Nx.as_type(:f32)
  end

  @doc """
  Forward pass to generate past_key_values for a batch.

  Returns a tensor with shape:
  `(batch_size, num_layers, 2, num_heads, num_virtual_tokens, head_dim)`

  ## Parameters

  - `encoder` - The PrefixEncoder struct
  - `batch_size` - Number of samples in the batch

  ## Examples

      past_key_values = Encoder.forward(encoder, 4)
      # => tensor with shape {4, num_layers, 2, num_heads, num_tokens, head_dim}

  """
  @spec forward(t(), pos_integer()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = encoder, batch_size) do
    config = encoder.config

    # Get the prefix embeddings
    prefix_embeddings =
      if encoder.use_projection do
        forward_with_projection(
          encoder.embedding,
          encoder.transform.w1,
          encoder.transform.b1,
          encoder.transform.w2,
          encoder.transform.b2
        )
      else
        encoder.embedding
      end

    # Reshape to (batch, num_layers, 2, num_heads, num_tokens, head_dim)
    reshape_to_past_key_values(
      prefix_embeddings,
      batch_size,
      config.num_layers,
      config.num_attention_heads,
      config.num_virtual_tokens,
      config.token_dim
    )
  end

  defnp forward_with_projection(embedding, w1, b1, w2, b2) do
    # embedding: (num_tokens, token_dim)
    # MLP: Linear -> Tanh -> Linear
    hidden = Nx.dot(embedding, w1) + b1
    hidden = Nx.tanh(hidden)
    Nx.dot(hidden, w2) + b2
  end

  # Use regular function since reshape needs runtime-known shapes
  defp reshape_to_past_key_values(
         embeddings,
         batch_size,
         num_layers,
         num_attention_heads,
         num_virtual_tokens,
         token_dim
       ) do
    head_dim = div(token_dim, num_attention_heads)

    # embeddings: (num_tokens, num_layers * 2 * token_dim)
    # Reshape to (num_tokens, num_layers, 2, num_heads, head_dim)
    reshaped =
      Nx.reshape(embeddings, {num_virtual_tokens, num_layers, 2, num_attention_heads, head_dim})

    # Transpose to (num_layers, 2, num_heads, num_tokens, head_dim)
    transposed = Nx.transpose(reshaped, axes: [1, 2, 3, 0, 4])

    # Broadcast to batch: (batch, num_layers, 2, num_heads, num_tokens, head_dim)
    Nx.broadcast(
      transposed,
      {batch_size, num_layers, 2, num_attention_heads, num_virtual_tokens, head_dim}
    )
  end

  @doc """
  Returns a map of all trainable parameters.

  ## Examples

      params = Encoder.get_trainable_params(encoder)
      # Without projection: %{"embedding" => tensor}
      # With projection: %{"embedding" => ..., "transform.w1" => ..., ...}

  """
  @spec get_trainable_params(t()) :: %{String.t() => Nx.Tensor.t()}
  def get_trainable_params(%__MODULE__{use_projection: false} = encoder) do
    %{"embedding" => encoder.embedding}
  end

  def get_trainable_params(%__MODULE__{use_projection: true} = encoder) do
    %{
      "embedding" => encoder.embedding,
      "transform.w1" => encoder.transform.w1,
      "transform.b1" => encoder.transform.b1,
      "transform.w2" => encoder.transform.w2,
      "transform.b2" => encoder.transform.b2
    }
  end

  @doc """
  Updates the encoder with new parameter values.

  ## Examples

      new_params = %{"embedding" => updated_embedding}
      updated_encoder = Encoder.set_trainable_params(encoder, new_params)

  """
  @spec set_trainable_params(t(), %{String.t() => Nx.Tensor.t()}) :: t()
  def set_trainable_params(%__MODULE__{use_projection: false} = encoder, params) do
    %{encoder | embedding: Map.get(params, "embedding", encoder.embedding)}
  end

  def set_trainable_params(%__MODULE__{use_projection: true} = encoder, params) do
    updated_transform = %{
      w1: Map.get(params, "transform.w1", encoder.transform.w1),
      b1: Map.get(params, "transform.b1", encoder.transform.b1),
      w2: Map.get(params, "transform.w2", encoder.transform.w2),
      b2: Map.get(params, "transform.b2", encoder.transform.b2)
    }

    %{
      encoder
      | embedding: Map.get(params, "embedding", encoder.embedding),
        transform: updated_transform
    }
  end

  @doc """
  Returns the total number of trainable parameters.

  ## Examples

      count = Encoder.param_count(encoder)

  """
  @spec param_count(t()) :: non_neg_integer()
  def param_count(%__MODULE__{} = encoder) do
    encoder
    |> get_trainable_params()
    |> Enum.reduce(0, fn {_name, tensor}, acc ->
      acc + Nx.size(tensor)
    end)
  end
end
