defmodule HfPeftEx.Tuners.Lora.Layer do
  @moduledoc """
  LoRA layer that wraps a base layer with low-rank adapters.

  The layer adds trainable low-rank matrices A and B such that:

      output = base_output + (x @ A^T @ B^T) * scaling

  Where:
  - A is initialized with Kaiming uniform initialization
  - B is initialized to zeros
  - scaling = lora_alpha / r (or lora_alpha / sqrt(r) with RSLoRA)

  ## Example

      layer = Layer.new(
        in_features: 1024,
        out_features: 1024,
        r: 8,
        lora_alpha: 16
      )
      
      # Forward pass adds LoRA contribution
      result = Layer.forward(layer, input, base_output)
      
      # Merge for inference
      {merged_layer, new_weight} = Layer.merge(layer, base_weight)

  """

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Dora

  @type t :: %__MODULE__{
          r: pos_integer(),
          lora_alpha: number(),
          lora_A: Nx.Tensor.t(),
          lora_B: Nx.Tensor.t(),
          scaling: float(),
          merged: boolean(),
          dropout: float(),
          in_features: pos_integer(),
          out_features: pos_integer(),
          use_rslora: boolean(),
          use_dora: boolean(),
          dora_magnitude: Nx.Tensor.t() | nil,
          dora_weight_norm: Nx.Tensor.t() | nil
        }

  defstruct [
    :r,
    :lora_alpha,
    :lora_A,
    :lora_B,
    :scaling,
    :in_features,
    :out_features,
    merged: false,
    dropout: 0.0,
    use_rslora: false,
    use_dora: false,
    dora_magnitude: nil,
    dora_weight_norm: nil
  ]

  @doc """
  Creates a new LoRA layer with the specified dimensions.

  ## Options

  - `:in_features` (required) - Input dimension
  - `:out_features` (required) - Output dimension  
  - `:r` - Rank (default: 8)
  - `:lora_alpha` - Scaling factor (default: 8)
  - `:dropout` - Dropout probability (default: 0.0)
  - `:use_rslora` - Use rank-stabilized LoRA scaling (default: false)
  - `:use_dora` - Enable DoRA scaling (default: false)
  - `:config` - LoraConfig struct to extract parameters from

  ## Examples

      # Basic usage
      layer = Layer.new(in_features: 1024, out_features: 512, r: 8)

      # With config
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      layer = Layer.new(in_features: 1024, out_features: 512, config: config)

  """
  @spec new(keyword()) :: t()
  def new(opts) do
    # Extract from config if provided
    opts = apply_config(opts)

    in_features = Keyword.fetch!(opts, :in_features)
    out_features = Keyword.fetch!(opts, :out_features)
    r = Keyword.get(opts, :r, 8)
    lora_alpha = Keyword.get(opts, :lora_alpha, 8)
    dropout = Keyword.get(opts, :dropout, 0.0)
    use_rslora = Keyword.get(opts, :use_rslora, false)
    use_dora = Keyword.get(opts, :use_dora, false)

    # Compute scaling factor
    scaling = compute_scaling(lora_alpha, r, use_rslora)

    # Initialize matrices
    # lora_A: r × in_features, initialized with Kaiming uniform
    # lora_B: out_features × r, initialized to zeros
    # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
    lora_A = init_lora_a(r, in_features)
    lora_B = init_lora_b(out_features, r)

    %__MODULE__{
      r: r,
      lora_alpha: lora_alpha,
      lora_A: lora_A,
      lora_B: lora_B,
      scaling: scaling,
      merged: false,
      dropout: dropout,
      in_features: in_features,
      out_features: out_features,
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
        |> Keyword.put_new(:dropout, config.lora_dropout)
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

  defp init_lora_a(r, in_features) do
    # Kaiming uniform initialization
    # fan_in = in_features for the LoRA A matrix
    bound = :math.sqrt(5) / :math.sqrt(in_features)
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.uniform(key, -bound, bound, shape: {r, in_features})
    tensor
  end

  defp init_lora_b(out_features, r) do
    # Initialize to zeros
    Nx.broadcast(Nx.tensor(0.0), {out_features, r})
  end

  @doc """
  Computes the delta weight: B @ A * scaling.

  This is the weight update that LoRA adds to the base layer.
  """
  @spec get_delta_weight(t()) :: Nx.Tensor.t()
  def get_delta_weight(%__MODULE__{} = layer) do
    # B @ A gives out_features × in_features
    Nx.dot(layer.lora_B, layer.lora_A)
    |> Nx.multiply(layer.scaling)
  end

  @doc """
  Forward pass through the LoRA layer.

  When merged, returns the base output unchanged.
  When not merged, adds the LoRA contribution:

      output = base_output + (x @ A^T @ B^T) * scaling

  ## Options

  - `:training` - Whether in training mode (enables dropout). Default: false
  - `:weight` - Base weight tensor (required when `use_dora` is true)
  """
  @spec forward(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward(layer, x, base_output, opts \\ [])

  def forward(%__MODULE__{merged: true}, _x, base_output, _opts) do
    base_output
  end

  def forward(%__MODULE__{use_dora: true} = layer, x, base_output, opts) do
    weight = Keyword.fetch!(opts, :weight)
    training = Keyword.get(opts, :training, false)

    # Apply dropout if training
    x = maybe_apply_dropout(x, layer.dropout, training)
    base_result = if training and layer.dropout > 0.0, do: nil, else: base_output

    magnitude = dora_magnitude(layer, weight)
    dora = %Dora{magnitude: magnitude, fan_in_fan_out: false}

    dora_delta =
      Dora.forward(dora, x, layer.lora_A, layer.lora_B, layer.scaling, weight, base_result)

    Nx.add(base_output, dora_delta)
  end

  def forward(%__MODULE__{} = layer, x, base_output, opts) do
    training = Keyword.get(opts, :training, false)

    # Apply dropout if training
    x = maybe_apply_dropout(x, layer.dropout, training)

    # Compute LoRA contribution: x @ A^T @ B^T * scaling
    # x: batch × in_features
    # A: r × in_features, so A^T: in_features × r
    # B: out_features × r, so B^T: r × out_features
    # Result: batch × out_features
    lora_output =
      x
      |> Nx.dot(Nx.transpose(layer.lora_A))
      |> Nx.dot(Nx.transpose(layer.lora_B))
      |> Nx.multiply(layer.scaling)

    Nx.add(base_output, lora_output)
  end

  defp maybe_apply_dropout(x, dropout, training) when training and dropout > 0.0 do
    key = Nx.Random.key(System.system_time())
    {mask, _key} = Nx.Random.uniform(key, shape: Nx.shape(x))
    # Scale by 1/(1-p) where p is dropout probability
    scale = 1.0 / (1.0 - dropout)
    keep_mask = Nx.greater(mask, dropout)
    x |> Nx.multiply(keep_mask) |> Nx.multiply(scale)
  end

  defp maybe_apply_dropout(x, _dropout, _training), do: x

  @doc """
  Merges the LoRA weights into the base weight.

  Returns {updated_layer, new_weight} where the layer is marked as merged
  and new_weight = base_weight + delta_weight.
  """
  @spec merge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = layer, base_weight) do
    {layer, base_weight}
  end

  def merge(%__MODULE__{} = layer, base_weight) do
    delta = get_delta_weight(layer)

    if layer.use_dora do
      magnitude = dora_magnitude(layer, base_weight)
      weight_norm = Dora.get_weight_norm(base_weight, delta, 1.0, fan_in_fan_out: false)
      dora_factor = dora_factor(magnitude, weight_norm)
      new_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {%{layer | merged: true, dora_magnitude: magnitude, dora_weight_norm: weight_norm},
       new_weight}
    else
      new_weight = Nx.add(base_weight, delta)
      {%{layer | merged: true}, new_weight}
    end
  end

  @doc """
  Unmerges the LoRA weights from the base weight.

  Returns {updated_layer, new_weight} where the layer is marked as unmerged
  and new_weight = base_weight - delta_weight.
  """
  @spec unmerge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = layer, base_weight) do
    {layer, base_weight}
  end

  def unmerge(%__MODULE__{} = layer, base_weight) do
    delta = get_delta_weight(layer)

    if layer.use_dora do
      magnitude =
        case layer.dora_magnitude do
          %Nx.Tensor{} = mag -> mag
          nil -> raise ArgumentError, "missing DoRA magnitude; merge before unmerge"
        end

      weight_norm =
        layer.dora_weight_norm ||
          raise ArgumentError, "missing DoRA weight norm; merge before unmerge"

      dora_factor = dora_factor(magnitude, weight_norm)
      new_weight = base_weight |> Nx.divide(dora_factor) |> Nx.subtract(delta)
      {%{layer | merged: false, dora_weight_norm: nil}, new_weight}
    else
      new_weight = Nx.subtract(base_weight, delta)
      {%{layer | merged: false}, new_weight}
    end
  end

  defp dora_magnitude(%__MODULE__{dora_magnitude: %Nx.Tensor{} = magnitude}, _weight) do
    magnitude
  end

  defp dora_magnitude(_layer, weight) do
    Dora.init_magnitude(weight)
  end

  defp dora_factor(magnitude, weight_norm) do
    dora_factor = Nx.divide(magnitude, weight_norm)
    {out_features} = Nx.shape(dora_factor)
    Nx.reshape(dora_factor, {out_features, 1})
  end
end
