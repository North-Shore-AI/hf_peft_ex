defmodule HfPeftEx.Tuners.Lora.Linear do
  @moduledoc """
  LoRA wrapper for Linear (dense) layers.

  This module provides LoRA (Low-Rank Adaptation) for dense/linear layers,
  supporting the `fan_in_fan_out` transposition mode where weights are stored
  as `{in_features, out_features}` instead of `{out_features, in_features}`.

  ## Example

      config = LoraConfig.new(r: 8, lora_alpha: 16)
      linear = Linear.new(1024, 512, config: config)
      
      # Forward pass with base layer output
      result = Linear.forward(linear, input, base_output)
      
      # Merge for inference
      {merged, new_weight} = Linear.merge(linear, base_weight)

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
          fan_in_fan_out: boolean(),
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
    fan_in_fan_out: false,
    use_dora: false,
    dora_magnitude: nil,
    dora_weight_norm: nil
  ]

  @doc """
  Creates a new Linear LoRA layer.

  ## Arguments

    * `in_features` - Input dimension
    * `out_features` - Output dimension
    * `opts` - Options

  ## Options

    * `:r` - Rank (default: 8)
    * `:lora_alpha` - Scaling factor (default: 8)
    * `:dropout` - Dropout probability (default: 0.0)
    * `:use_rslora` - Use rank-stabilized LoRA scaling (default: false)
    * `:fan_in_fan_out` - Weight stored as {in, out} (default: false)
    * `:use_dora` - Enable DoRA scaling (default: false)
    * `:config` - LoraConfig struct to extract parameters from

  ## Examples

      linear = Linear.new(1024, 512, r: 8, lora_alpha: 16)

      config = LoraConfig.new(r: 16, lora_alpha: 32)
      linear = Linear.new(1024, 512, config: config)

  """
  @spec new(pos_integer(), pos_integer(), keyword()) :: t()
  def new(in_features, out_features, opts \\ []) do
    opts = apply_config(opts)

    r = Keyword.get(opts, :r, 8)
    lora_alpha = Keyword.get(opts, :lora_alpha, 8)
    dropout = Keyword.get(opts, :dropout, 0.0)
    use_rslora = Keyword.get(opts, :use_rslora, false)
    fan_in_fan_out = Keyword.get(opts, :fan_in_fan_out, false)
    use_dora = Keyword.get(opts, :use_dora, false)

    scaling = compute_scaling(lora_alpha, r, use_rslora)

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
      fan_in_fan_out: fan_in_fan_out,
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
        |> Keyword.put_new(:fan_in_fan_out, config.fan_in_fan_out)
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
    bound = :math.sqrt(5) / :math.sqrt(in_features)
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.uniform(key, -bound, bound, shape: {r, in_features})
    tensor
  end

  defp init_lora_b(out_features, r) do
    Nx.broadcast(Nx.tensor(0.0), {out_features, r})
  end

  @doc """
  Computes the delta weight: B @ A * scaling.

  When `fan_in_fan_out` is true, the result is transposed.
  """
  @spec get_delta_weight(t()) :: Nx.Tensor.t()
  def get_delta_weight(%__MODULE__{} = linear) do
    delta = Nx.dot(linear.lora_B, linear.lora_A) |> Nx.multiply(linear.scaling)

    if linear.fan_in_fan_out do
      Nx.transpose(delta)
    else
      delta
    end
  end

  @doc """
  Forward pass through the Linear LoRA layer.

  When merged, returns the base output unchanged.
  When not merged, adds the LoRA contribution.

  ## Options

    * `:training` - Whether in training mode (enables dropout). Default: false
    * `:weight` - Base weight tensor (required when `use_dora` is true)

  """
  @spec forward(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward(linear, x, base_output, opts \\ [])

  def forward(%__MODULE__{merged: true}, _x, base_output, _opts) do
    base_output
  end

  def forward(%__MODULE__{use_dora: true} = linear, x, base_output, opts) do
    weight = Keyword.fetch!(opts, :weight)
    training = Keyword.get(opts, :training, false)
    x = maybe_apply_dropout(x, linear.dropout, training)
    base_result = if training and linear.dropout > 0.0, do: nil, else: base_output

    magnitude = dora_magnitude(linear, weight)
    dora = %Dora{magnitude: magnitude, fan_in_fan_out: linear.fan_in_fan_out}

    dora_delta =
      Dora.forward(dora, x, linear.lora_A, linear.lora_B, linear.scaling, weight, base_result)

    Nx.add(base_output, dora_delta)
  end

  def forward(%__MODULE__{} = linear, x, base_output, opts) do
    training = Keyword.get(opts, :training, false)
    x = maybe_apply_dropout(x, linear.dropout, training)

    lora_output =
      x
      |> Nx.dot(Nx.transpose(linear.lora_A))
      |> Nx.dot(Nx.transpose(linear.lora_B))
      |> Nx.multiply(linear.scaling)

    Nx.add(base_output, lora_output)
  end

  defp maybe_apply_dropout(x, dropout, training) when training and dropout > 0.0 do
    key = Nx.Random.key(System.system_time())
    {mask, _key} = Nx.Random.uniform(key, shape: Nx.shape(x))
    scale = 1.0 / (1.0 - dropout)
    keep_mask = Nx.greater(mask, dropout)
    x |> Nx.multiply(keep_mask) |> Nx.multiply(scale)
  end

  defp maybe_apply_dropout(x, _dropout, _training), do: x

  @doc """
  Merges the LoRA weights into the base weight.

  Returns `{updated_linear, new_weight}` where the linear is marked as merged
  and `new_weight = base_weight + delta_weight`.
  """
  @spec merge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = linear, base_weight) do
    {linear, base_weight}
  end

  def merge(%__MODULE__{} = linear, base_weight) do
    delta = get_delta_weight(linear)

    if linear.use_dora do
      magnitude = dora_magnitude(linear, base_weight)
      lora_weight = maybe_transpose(delta, linear.fan_in_fan_out)

      weight_norm =
        Dora.get_weight_norm(base_weight, lora_weight, 1.0, fan_in_fan_out: linear.fan_in_fan_out)

      dora_factor = dora_factor(magnitude, weight_norm, linear.fan_in_fan_out)
      new_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {%{linear | merged: true, dora_magnitude: magnitude, dora_weight_norm: weight_norm},
       new_weight}
    else
      new_weight = Nx.add(base_weight, delta)
      {%{linear | merged: true}, new_weight}
    end
  end

  @doc """
  Unmerges the LoRA weights from the base weight.

  Returns `{updated_linear, new_weight}` where the linear is marked as unmerged
  and `new_weight = base_weight - delta_weight`.
  """
  @spec unmerge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = linear, base_weight) do
    {linear, base_weight}
  end

  def unmerge(%__MODULE__{} = linear, base_weight) do
    delta = get_delta_weight(linear)

    if linear.use_dora do
      magnitude =
        case linear.dora_magnitude do
          %Nx.Tensor{} = mag -> mag
          nil -> raise ArgumentError, "missing DoRA magnitude; merge before unmerge"
        end

      weight_norm =
        linear.dora_weight_norm ||
          raise ArgumentError, "missing DoRA weight norm; merge before unmerge"

      dora_factor = dora_factor(magnitude, weight_norm, linear.fan_in_fan_out)
      new_weight = base_weight |> Nx.divide(dora_factor) |> Nx.subtract(delta)
      {%{linear | merged: false, dora_weight_norm: nil}, new_weight}
    else
      new_weight = Nx.subtract(base_weight, delta)
      {%{linear | merged: false}, new_weight}
    end
  end

  defp dora_magnitude(%__MODULE__{dora_magnitude: %Nx.Tensor{} = magnitude}, _weight) do
    magnitude
  end

  defp dora_magnitude(%__MODULE__{fan_in_fan_out: fan_in_fan_out}, weight) do
    weight
    |> maybe_transpose(fan_in_fan_out)
    |> Dora.init_magnitude()
  end

  defp dora_factor(magnitude, weight_norm, fan_in_fan_out) do
    dora_factor = Nx.divide(magnitude, weight_norm)
    {out_features} = Nx.shape(dora_factor)
    dora_factor = Nx.reshape(dora_factor, {out_features, 1})

    if fan_in_fan_out do
      Nx.transpose(dora_factor)
    else
      dora_factor
    end
  end

  defp maybe_transpose(weight, true), do: Nx.transpose(weight)
  defp maybe_transpose(weight, false), do: weight
end
