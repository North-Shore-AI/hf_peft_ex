defmodule HfPeftEx.Tuners.Lora.Embedding do
  @moduledoc """
  LoRA wrapper for Embedding layers.

  This module provides LoRA (Low-Rank Adaptation) for embedding layers.
  Unlike Linear layers, embeddings use `lora_embedding_A` and `lora_embedding_B`
  as parameters (not linear layers), and always use transposition in delta weight.

  ## Example

      config = LoraConfig.new(r: 8, lora_alpha: 16)
      embed = Embedding.new(10_000, 768, config: config)
      
      # Forward pass with base layer output
      result = Embedding.forward(embed, indices, base_output)
      
      # Merge for inference
      {merged, new_weight} = Embedding.merge(embed, base_weight)

  """

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Dora

  @type t :: %__MODULE__{
          r: pos_integer(),
          lora_alpha: number(),
          lora_embedding_A: Nx.Tensor.t(),
          lora_embedding_B: Nx.Tensor.t(),
          scaling: float(),
          merged: boolean(),
          num_embeddings: pos_integer(),
          embedding_dim: pos_integer(),
          use_rslora: boolean(),
          use_dora: boolean(),
          dora_magnitude: Nx.Tensor.t() | nil,
          dora_weight_norm: Nx.Tensor.t() | nil
        }

  defstruct [
    :r,
    :lora_alpha,
    :lora_embedding_A,
    :lora_embedding_B,
    :scaling,
    :num_embeddings,
    :embedding_dim,
    merged: false,
    use_rslora: false,
    use_dora: false,
    dora_magnitude: nil,
    dora_weight_norm: nil
  ]

  @doc """
  Creates a new Embedding LoRA layer.

  ## Arguments

    * `num_embeddings` - Size of the vocabulary
    * `embedding_dim` - Dimension of embeddings
    * `opts` - Options

  ## Options

    * `:r` - Rank (default: 8)
    * `:lora_alpha` - Scaling factor (default: 8)
    * `:use_rslora` - Use rank-stabilized LoRA scaling (default: false)
    * `:use_dora` - Enable DoRA scaling (default: false)
    * `:config` - LoraConfig struct to extract parameters from

  ## Examples

      embed = Embedding.new(10_000, 768, r: 8, lora_alpha: 16)

      config = LoraConfig.new(r: 16, lora_alpha: 32)
      embed = Embedding.new(10_000, 768, config: config)

  """
  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  def new(num_embeddings, embedding_dim, opts \\ []) do
    opts = apply_config(opts)

    r = Keyword.get(opts, :r, 8)
    lora_alpha = Keyword.get(opts, :lora_alpha, 8)
    use_rslora = Keyword.get(opts, :use_rslora, false)
    use_dora = Keyword.get(opts, :use_dora, false)

    scaling = compute_scaling(lora_alpha, r, use_rslora)

    # Initialize with random values (unlike Linear which uses zeros for B)
    key = Nx.Random.key(System.system_time())
    # credo:disable-for-lines:4 Credo.Check.Readability.VariableNames
    {lora_embedding_A, key} = Nx.Random.normal(key, shape: {r, num_embeddings})
    {lora_embedding_B, _key} = Nx.Random.normal(key, shape: {embedding_dim, r})

    %__MODULE__{
      r: r,
      lora_alpha: lora_alpha,
      lora_embedding_A: lora_embedding_A,
      lora_embedding_B: lora_embedding_B,
      scaling: scaling,
      merged: false,
      num_embeddings: num_embeddings,
      embedding_dim: embedding_dim,
      use_rslora: use_rslora,
      use_dora: use_dora,
      dora_magnitude: nil,
      dora_weight_norm: nil
    }
  end

  defp apply_config(opts) do
    case Keyword.get(opts, :config) do
      nil ->
        opts

      %LoraConfig{} = config ->
        opts
        |> Keyword.put_new(:r, config.r)
        |> Keyword.put_new(:lora_alpha, config.lora_alpha)
        |> Keyword.put_new(:use_rslora, config.use_rslora)
        |> Keyword.put_new(:use_dora, config.use_dora)
    end
  end

  defp compute_scaling(lora_alpha, r, use_rslora) do
    if use_rslora do
      lora_alpha / :math.sqrt(r)
    else
      lora_alpha / r
    end
  end

  @doc """
  Computes the delta weight: (B @ A)^T * scaling.

  For embeddings, the result is always transposed to match
  the weight shape {num_embeddings, embedding_dim}.
  """
  @spec get_delta_weight(t()) :: Nx.Tensor.t()
  def get_delta_weight(%__MODULE__{} = embed) do
    # B @ A = {embedding_dim, r} @ {r, num_embeddings} = {embedding_dim, num_embeddings}
    # Transpose to get {num_embeddings, embedding_dim}
    Nx.dot(embed.lora_embedding_B, embed.lora_embedding_A)
    |> Nx.transpose()
    |> Nx.multiply(embed.scaling)
  end

  @doc """
  Looks up embedding vectors for given indices.

  This performs the equivalent of `torch.nn.functional.embedding`.
  """
  @spec embed_lookup(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def embed_lookup(indices, weight) do
    # indices: {batch, seq_len}
    # weight: {num_embeddings, embedding_dim}
    # result: {batch, seq_len, embedding_dim}
    Nx.take(weight, indices, axis: 0)
  end

  @doc """
  Forward pass through the Embedding LoRA layer.

  When merged, returns the base output unchanged.
  When not merged, adds the LoRA contribution.

  ## Options

    * `:weight` - Base weight tensor (required when `use_dora` is true)

  ## Examples

      result = Embedding.forward(embed, indices, base_output)

  """
  @spec forward(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward(embed, indices, base_output, opts \\ [])

  def forward(%__MODULE__{merged: true}, _indices, base_output, _opts) do
    base_output
  end

  def forward(%__MODULE__{use_dora: true} = embed, indices, base_output, opts) do
    weight = Keyword.fetch!(opts, :weight)

    # A^T: {num_embeddings, r}
    # Lookup: {batch, seq_len, r}
    # credo:disable-for-next-line Credo.Check.Readability.VariableNames
    after_A = embed_lookup(indices, Nx.transpose(embed.lora_embedding_A))

    # B^T: {r, embedding_dim}
    # after_A @ B^T: {batch, seq_len, embedding_dim}
    lora_output =
      Nx.dot(after_A, Nx.transpose(embed.lora_embedding_B))
      |> Nx.multiply(embed.scaling)

    magnitude = dora_magnitude(embed, weight)
    lora_weight = Nx.dot(embed.lora_embedding_B, embed.lora_embedding_A)
    weight_norm = Dora.get_weight_norm(weight, lora_weight, embed.scaling, fan_in_fan_out: true)
    mag_norm_scale = Nx.divide(magnitude, weight_norm) |> Nx.reshape({1, 1, embed.embedding_dim})

    base_output
    |> Nx.add(lora_output)
    |> Nx.multiply(mag_norm_scale)
  end

  def forward(%__MODULE__{} = embed, indices, base_output, _opts) do
    # A^T: {num_embeddings, r}
    # Lookup: {batch, seq_len, r}
    # credo:disable-for-next-line Credo.Check.Readability.VariableNames
    after_A = embed_lookup(indices, Nx.transpose(embed.lora_embedding_A))

    # B^T: {r, embedding_dim}
    # after_A @ B^T: {batch, seq_len, embedding_dim}
    lora_output =
      Nx.dot(after_A, Nx.transpose(embed.lora_embedding_B))
      |> Nx.multiply(embed.scaling)

    Nx.add(base_output, lora_output)
  end

  @doc """
  Merges the LoRA weights into the base weight.

  Returns `{updated_embed, new_weight}` where the embed is marked as merged
  and `new_weight = base_weight + delta_weight`.
  """
  @spec merge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = embed, base_weight) do
    {embed, base_weight}
  end

  def merge(%__MODULE__{} = embed, base_weight) do
    delta = get_delta_weight(embed)

    if embed.use_dora do
      magnitude = dora_magnitude(embed, base_weight)

      weight_norm =
        Dora.get_weight_norm(base_weight, Nx.transpose(delta), 1.0, fan_in_fan_out: true)

      dora_factor = dora_factor(magnitude, weight_norm)
      new_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {%{embed | merged: true, dora_magnitude: magnitude, dora_weight_norm: weight_norm},
       new_weight}
    else
      new_weight = Nx.add(base_weight, delta)
      {%{embed | merged: true}, new_weight}
    end
  end

  @doc """
  Unmerges the LoRA weights from the base weight.

  Returns `{updated_embed, new_weight}` where the embed is marked as unmerged
  and `new_weight = base_weight - delta_weight`.
  """
  @spec unmerge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = embed, base_weight) do
    {embed, base_weight}
  end

  def unmerge(%__MODULE__{} = embed, base_weight) do
    delta = get_delta_weight(embed)

    if embed.use_dora do
      magnitude =
        case embed.dora_magnitude do
          %Nx.Tensor{} = mag -> mag
          nil -> raise ArgumentError, "missing DoRA magnitude; merge before unmerge"
        end

      weight_norm =
        embed.dora_weight_norm ||
          raise ArgumentError, "missing DoRA weight norm; merge before unmerge"

      dora_factor = dora_factor(magnitude, weight_norm)
      new_weight = base_weight |> Nx.divide(dora_factor) |> Nx.subtract(delta)
      {%{embed | merged: false, dora_weight_norm: nil}, new_weight}
    else
      new_weight = Nx.subtract(base_weight, delta)
      {%{embed | merged: false}, new_weight}
    end
  end

  defp dora_magnitude(%__MODULE__{dora_magnitude: %Nx.Tensor{} = magnitude}, _weight) do
    magnitude
  end

  defp dora_magnitude(_embed, weight) do
    weight
    |> Nx.transpose()
    |> Dora.init_magnitude()
  end

  defp dora_factor(magnitude, weight_norm) do
    dora_factor = Nx.divide(magnitude, weight_norm)
    {embedding_dim} = Nx.shape(dora_factor)
    Nx.reshape(dora_factor, {1, embedding_dim})
  end
end
