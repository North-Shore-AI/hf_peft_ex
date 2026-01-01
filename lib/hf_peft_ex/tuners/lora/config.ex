defmodule HfPeftEx.Tuners.Lora.Config do
  @moduledoc """
  Configuration class for LoRA (Low-Rank Adaptation).

  LoRA freezes the pre-trained model weights and injects trainable low-rank
  decomposition matrices into each layer, greatly reducing the number of
  trainable parameters for downstream tasks.

  ## Key Parameters

  - `:r` - LoRA rank (dimension of the low-rank matrices)
  - `:lora_alpha` - Scaling factor for LoRA
  - `:lora_dropout` - Dropout probability for LoRA layers
  - `:target_modules` - Which modules to apply LoRA to

  ## Example

      config = HfPeftEx.Tuners.Lora.Config.new(
        r: 16,
        lora_alpha: 32,
        target_modules: ["q_proj", "v_proj"],
        task_type: :causal_lm
      )

  ## LoRA Math

  For a pretrained weight matrix W₀, LoRA represents updates as:

      W = W₀ + BA

  Where B ∈ ℝᵈˣʳ and A ∈ ℝʳˣᵏ with rank r << min(d,k)

  The scaling factor is: `lora_alpha / r`

  """

  alias HfPeftEx.{PeftType, TaskType}

  @type bias :: :none | :all | :lora_only

  @type init_method ::
          boolean()
          | :gaussian
          | :eva
          | :olora
          | :pissa
          | :corda
          | :loftq
          | :orthogonal

  @type t :: %__MODULE__{
          # Base config fields
          peft_type: PeftType.t(),
          task_type: TaskType.t() | nil,
          base_model_name_or_path: String.t() | nil,
          revision: String.t() | nil,
          inference_mode: boolean(),
          peft_version: String.t() | nil,
          # LoRA specific fields
          r: pos_integer(),
          target_modules: [String.t()] | String.t() | nil,
          exclude_modules: [String.t()] | String.t() | nil,
          lora_alpha: number(),
          lora_dropout: float(),
          fan_in_fan_out: boolean(),
          bias: bias(),
          use_rslora: boolean(),
          modules_to_save: [String.t()] | nil,
          init_lora_weights: init_method(),
          layers_to_transform: [integer()] | integer() | nil,
          layers_pattern: [String.t()] | String.t() | nil,
          rank_pattern: map(),
          alpha_pattern: map(),
          use_dora: boolean(),
          lora_bias: boolean()
        }

  @enforce_keys []
  defstruct peft_type: :lora,
            task_type: nil,
            base_model_name_or_path: nil,
            revision: nil,
            inference_mode: false,
            peft_version: nil,
            # LoRA defaults
            r: 8,
            target_modules: nil,
            exclude_modules: nil,
            lora_alpha: 8,
            lora_dropout: 0.0,
            fan_in_fan_out: false,
            bias: :none,
            use_rslora: false,
            modules_to_save: nil,
            init_lora_weights: true,
            layers_to_transform: nil,
            layers_pattern: nil,
            rank_pattern: %{},
            alpha_pattern: %{},
            use_dora: false,
            lora_bias: false

  @doc """
  Creates a new LoRA configuration.

  ## Options

  ### Required (or recommended)
  - `:r` - LoRA rank (default: `8`). Higher rank = more parameters but better fit
  - `:lora_alpha` - Scaling factor (default: `8`). Often set to 2× the rank

  ### Target Modules
  - `:target_modules` - Module names to apply LoRA. Can be:
    - List of strings: `["q_proj", "v_proj"]`
    - Regex string: `".*attention.*"`
    - `"all-linear"` to target all linear layers
  - `:exclude_modules` - Modules to exclude from LoRA

  ### Training Configuration
  - `:task_type` - Task type (`:causal_lm`, `:seq_cls`, etc.)
  - `:lora_dropout` - Dropout rate (default: `0.0`)
  - `:bias` - Bias configuration: `:none`, `:all`, or `:lora_only`
  - `:modules_to_save` - Extra modules to train and save

  ### Advanced Options
  - `:use_rslora` - Use Rank-Stabilized LoRA scaling (default: `false`)
  - `:use_dora` - Enable DoRA (Weight-Decomposed LoRA) (default: `false`)
  - `:init_lora_weights` - Initialization method (default: `true`)
  - `:rank_pattern` - Per-layer rank overrides
  - `:alpha_pattern` - Per-layer alpha overrides

  ## Examples

      # Basic configuration
      config = HfPeftEx.Tuners.Lora.Config.new(r: 16, lora_alpha: 32)

      # For a causal LM with specific target modules
      config = HfPeftEx.Tuners.Lora.Config.new(
        r: 8,
        lora_alpha: 16,
        target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type: :causal_lm,
        lora_dropout: 0.05
      )

      # With DoRA for improved performance
      config = HfPeftEx.Tuners.Lora.Config.new(
        r: 8,
        use_dora: true
      )

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version(), peft_type: :lora}
    validate!(config)
  end

  @doc """
  Validates a LoRA configuration.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    validate_rank!(config)
    validate_dropout!(config)
    validate_bias!(config)
    validate_task_type!(config)
    validate_dora_compatibility!(config)
    config
  end

  defp validate_rank!(%{r: r}) when r <= 0 do
    raise ArgumentError, "LoRA rank `r` must be positive, got: #{r}"
  end

  defp validate_rank!(_config), do: :ok

  defp validate_dropout!(%{lora_dropout: dropout}) when dropout < 0.0 or dropout >= 1.0 do
    raise ArgumentError, "lora_dropout must be in [0.0, 1.0), got: #{dropout}"
  end

  defp validate_dropout!(_config), do: :ok

  defp validate_bias!(%{bias: bias}) when bias not in [:none, :all, :lora_only] do
    raise ArgumentError, "bias must be :none, :all, or :lora_only, got: #{inspect(bias)}"
  end

  defp validate_bias!(_config), do: :ok

  defp validate_task_type!(%{task_type: nil}), do: :ok

  defp validate_task_type!(%{task_type: task_type}) do
    unless TaskType.valid?(task_type) do
      raise ArgumentError, "Invalid task_type: #{inspect(task_type)}"
    end
  end

  defp validate_dora_compatibility!(%{use_dora: true, layers_pattern: pattern})
       when not is_nil(pattern) do
    raise ArgumentError, "DoRA does not support layers_pattern"
  end

  defp validate_dora_compatibility!(_config), do: :ok

  @doc """
  Returns the LoRA scaling factor.

  When `use_rslora` is true, uses rank-stabilized scaling: `lora_alpha / sqrt(r)`
  Otherwise uses standard scaling: `lora_alpha / r`
  """
  @spec scaling(t()) :: float()
  def scaling(%__MODULE__{use_rslora: true, lora_alpha: alpha, r: r}) do
    alpha / :math.sqrt(r)
  end

  def scaling(%__MODULE__{lora_alpha: alpha, r: r}) do
    alpha / r
  end

  @doc """
  Converts the configuration to a map for JSON serialization.
  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = config) do
    config
    |> Map.from_struct()
    |> Map.new(fn
      {k, v} when is_atom(v) and not is_nil(v) and not is_boolean(v) ->
        {k, v |> Atom.to_string() |> String.upcase()}

      pair ->
        pair
    end)
  end

  @doc """
  Creates a configuration from a map (e.g., loaded from JSON).
  """
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    # Convert string keys to atoms
    map =
      Map.new(map, fn
        {k, v} when is_binary(k) ->
          key =
            try do
              String.to_existing_atom(k)
            rescue
              ArgumentError -> String.to_atom(k)
            end

          {key, v}

        pair ->
          pair
      end)

    # Convert specific fields
    map = convert_atom_field(map, :peft_type)
    map = convert_atom_field(map, :task_type)
    map = convert_bias_field(map)

    struct(__MODULE__, map)
  end

  defp convert_atom_field(map, key) do
    case Map.get(map, key) do
      nil -> map
      val when is_binary(val) -> Map.put(map, key, val |> String.downcase() |> String.to_atom())
      _ -> map
    end
  end

  defp convert_bias_field(map) do
    case Map.get(map, :bias) do
      nil -> map
      "none" -> Map.put(map, :bias, :none)
      "all" -> Map.put(map, :bias, :all)
      "lora_only" -> Map.put(map, :bias, :lora_only)
      val when is_atom(val) -> map
      _ -> map
    end
  end

  @doc """
  Saves the LoRA configuration to a directory.

  Creates an `adapter_config.json` file in the specified directory.
  """
  @spec save_pretrained(t(), String.t()) :: :ok | {:error, term()}
  def save_pretrained(%__MODULE__{} = config, save_directory) do
    File.mkdir_p!(save_directory)
    output_path = Path.join(save_directory, "adapter_config.json")

    json = config |> to_map() |> Jason.encode!(pretty: true)
    File.write(output_path, json)
  end

  @doc """
  Loads a LoRA configuration from a directory or file.
  """
  @spec from_pretrained(String.t()) :: {:ok, t()} | {:error, term()}
  def from_pretrained(path) do
    config_path =
      if File.dir?(path) do
        Path.join(path, "adapter_config.json")
      else
        path
      end

    with {:ok, content} <- File.read(config_path),
         {:ok, json} <- Jason.decode(content) do
      config = from_map(json)
      {:ok, config}
    end
  end

  @doc """
  Returns the trainable parameter count estimate for a given layer.

  For a layer with input dimension `in_features` and output dimension `out_features`,
  LoRA adds approximately `r * (in_features + out_features)` parameters.
  """
  @spec trainable_params(t(), pos_integer(), pos_integer()) :: pos_integer()
  def trainable_params(%__MODULE__{r: r, use_dora: use_dora}, in_features, out_features) do
    base_params = r * (in_features + out_features)

    if use_dora do
      # DoRA adds magnitude vector of size out_features
      base_params + out_features
    else
      base_params
    end
  end
end
