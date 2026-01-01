defmodule HfPeftEx.Tuners.Lora.Conv1d do
  @moduledoc """
  LoRA wrapper for 1D convolution layers.

  Uses low-rank matrices A and B to build a delta weight that is added
  to the base convolution weight. DoRA scaling is supported.
  """

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig

  @type t :: %__MODULE__{
          r: pos_integer(),
          lora_alpha: number(),
          lora_A: Nx.Tensor.t(),
          lora_B: Nx.Tensor.t(),
          scaling: float(),
          merged: boolean(),
          dropout: float(),
          in_channels: pos_integer(),
          out_channels: pos_integer(),
          kernel_size: pos_integer(),
          stride: pos_integer(),
          padding: :valid | :same | non_neg_integer(),
          dilation: pos_integer(),
          groups: pos_integer(),
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
    :in_channels,
    :out_channels,
    :kernel_size,
    :stride,
    :padding,
    :dilation,
    :groups,
    merged: false,
    dropout: 0.0,
    use_rslora: false,
    use_dora: false,
    dora_magnitude: nil,
    dora_weight_norm: nil
  ]

  @doc """
  Creates a new Conv1d LoRA layer.

  ## Options

    * `:r` - Rank (default: 8)
    * `:lora_alpha` - Scaling factor (default: 8)
    * `:dropout` - Dropout probability (default: 0.0)
    * `:use_rslora` - Use rank-stabilized scaling (default: false)
    * `:use_dora` - Enable DoRA scaling (default: false)
    * `:stride` - Convolution stride (default: 1)
    * `:padding` - Padding (`:valid`, `:same`, or integer) (default: :valid)
    * `:dilation` - Kernel dilation (default: 1)
    * `:groups` - Feature groups (default: 1)
    * `:config` - LoraConfig struct to extract parameters from
  """
  @spec new(pos_integer(), pos_integer(), pos_integer(), keyword()) :: t()
  def new(in_channels, out_channels, kernel_size, opts \\ []) do
    opts = apply_config(opts)

    r = Keyword.get(opts, :r, 8)
    lora_alpha = Keyword.get(opts, :lora_alpha, 8)
    dropout = Keyword.get(opts, :dropout, 0.0)
    use_rslora = Keyword.get(opts, :use_rslora, false)
    use_dora = Keyword.get(opts, :use_dora, false)
    stride = Keyword.get(opts, :stride, 1)
    padding = Keyword.get(opts, :padding, :valid)
    dilation = Keyword.get(opts, :dilation, 1)
    groups = Keyword.get(opts, :groups, 1)

    scaling = compute_scaling(lora_alpha, r, use_rslora)

    lora_a = init_lora_a(r, in_channels * kernel_size)
    lora_b = init_lora_b(out_channels, r)

    %__MODULE__{
      r: r,
      lora_alpha: lora_alpha,
      lora_A: lora_a,
      lora_B: lora_b,
      scaling: scaling,
      merged: false,
      dropout: dropout,
      in_channels: in_channels,
      out_channels: out_channels,
      kernel_size: kernel_size,
      stride: stride,
      padding: padding,
      dilation: dilation,
      groups: groups,
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

  defp init_lora_a(r, fan_in) do
    bound = :math.sqrt(5) / :math.sqrt(fan_in)
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.uniform(key, -bound, bound, shape: {r, fan_in})
    tensor
  end

  defp init_lora_b(out_channels, r) do
    Nx.broadcast(Nx.tensor(0.0), {out_channels, r})
  end

  @doc """
  Computes the delta weight for Conv1d.
  """
  @spec get_delta_weight(t()) :: Nx.Tensor.t()
  def get_delta_weight(%__MODULE__{} = conv) do
    conv
    |> lora_weight()
    |> Nx.multiply(conv.scaling)
  end

  @doc """
  Forward pass through Conv1d with LoRA (and optional DoRA).
  """
  @spec forward(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def forward(conv, x, base_output, opts \\ [])

  def forward(%__MODULE__{merged: true}, _x, base_output, _opts) do
    base_output
  end

  def forward(%__MODULE__{use_dora: true} = conv, x, base_output, opts) do
    weight = Keyword.fetch!(opts, :weight)
    training = Keyword.get(opts, :training, false)
    x = maybe_apply_dropout(x, conv.dropout, training)
    base_result = if training and conv.dropout > 0.0, do: nil, else: base_output

    lora_weight = lora_weight(conv)
    lora_output = conv_out(x, lora_weight, conv)
    base_result = base_result || conv_out(x, weight, conv)

    magnitude = dora_magnitude(conv, weight)
    weight_norm = weight_norm(weight, lora_weight, conv.scaling)
    mag_norm_scale = output_scale(magnitude, weight_norm, conv)

    dora_delta =
      mag_norm_scale
      |> Nx.subtract(1.0)
      |> Nx.multiply(base_result)
      |> Nx.add(Nx.multiply(mag_norm_scale, Nx.multiply(lora_output, conv.scaling)))

    Nx.add(base_output, dora_delta)
  end

  def forward(%__MODULE__{} = conv, x, base_output, opts) do
    training = Keyword.get(opts, :training, false)
    x = maybe_apply_dropout(x, conv.dropout, training)

    lora_output =
      x
      |> conv_out(lora_weight(conv), conv)
      |> Nx.multiply(conv.scaling)

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
  """
  @spec merge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def merge(%__MODULE__{merged: true} = conv, base_weight) do
    {conv, base_weight}
  end

  def merge(%__MODULE__{} = conv, base_weight) do
    delta = get_delta_weight(conv)

    if conv.use_dora do
      magnitude = dora_magnitude(conv, base_weight)
      weight_norm = weight_norm(base_weight, lora_weight(conv), conv.scaling)
      dora_factor = weight_scale(magnitude, weight_norm, conv)
      new_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {%{conv | merged: true, dora_magnitude: magnitude, dora_weight_norm: weight_norm},
       new_weight}
    else
      new_weight = Nx.add(base_weight, delta)
      {%{conv | merged: true}, new_weight}
    end
  end

  @doc """
  Unmerges the LoRA weights from the base weight.
  """
  @spec unmerge(t(), Nx.Tensor.t()) :: {t(), Nx.Tensor.t()}
  def unmerge(%__MODULE__{merged: false} = conv, base_weight) do
    {conv, base_weight}
  end

  def unmerge(%__MODULE__{} = conv, base_weight) do
    delta = get_delta_weight(conv)

    if conv.use_dora do
      magnitude =
        case conv.dora_magnitude do
          %Nx.Tensor{} = mag -> mag
          nil -> raise ArgumentError, "missing DoRA magnitude; merge before unmerge"
        end

      weight_norm =
        conv.dora_weight_norm ||
          raise ArgumentError, "missing DoRA weight norm; merge before unmerge"

      dora_factor = weight_scale(magnitude, weight_norm, conv)
      new_weight = base_weight |> Nx.divide(dora_factor) |> Nx.subtract(delta)
      {%{conv | merged: false, dora_weight_norm: nil}, new_weight}
    else
      new_weight = Nx.subtract(base_weight, delta)
      {%{conv | merged: false}, new_weight}
    end
  end

  defp lora_weight(%__MODULE__{} = conv) do
    Nx.dot(conv.lora_B, conv.lora_A)
    |> Nx.reshape({conv.out_channels, conv.in_channels, conv.kernel_size})
  end

  defp weight_norm(weight, lora_weight, scaling) do
    combined = weight |> Nx.add(Nx.multiply(lora_weight, scaling))

    combined
    |> Nx.multiply(combined)
    |> Nx.sum(axes: [1, 2])
    |> Nx.sqrt()
  end

  defp dora_magnitude(%__MODULE__{dora_magnitude: %Nx.Tensor{} = magnitude}, _weight) do
    magnitude
  end

  defp dora_magnitude(_conv, weight) do
    weight
    |> Nx.multiply(weight)
    |> Nx.sum(axes: [1, 2])
    |> Nx.sqrt()
  end

  defp weight_scale(magnitude, weight_norm, _conv) do
    {out_channels} = Nx.shape(magnitude)
    dora_factor = Nx.divide(magnitude, weight_norm)
    Nx.reshape(dora_factor, {out_channels, 1, 1})
  end

  defp output_scale(magnitude, weight_norm, _conv) do
    {out_channels} = Nx.shape(magnitude)
    mag_norm_scale = Nx.divide(magnitude, weight_norm)
    Nx.reshape(mag_norm_scale, {1, out_channels, 1})
  end

  defp conv_out(x, weight, conv) do
    Nx.conv(x, weight,
      strides: conv.stride,
      padding: conv.padding,
      kernel_dilation: conv.dilation,
      feature_group_size: conv.groups
    )
  end
end
