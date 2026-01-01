defmodule HfPeftEx.Tuners.Adalora.Config do
  @moduledoc """
  Configuration class for AdaLoRA (Adaptive Low-Rank Adaptation).

  AdaLoRA dynamically allocates rank budget across layers during training
  based on importance scores. Less important layers get pruned while
  important layers retain higher rank.

  ## Three Phases

  1. **Warmup (tinit)**: Pre-training phase with no rank pruning
  2. **Budgeting**: Rank is gradually reduced from `init_r` to `target_r`
  3. **Final (tfinal)**: Fine-tuning phase with frozen ranks

  ## Key Parameters

  - `:total_step` - Total training steps (required)
  - `:init_r` - Initial rank before pruning
  - `:target_r` - Target rank after pruning
  - `:tinit` - Warmup steps before pruning begins
  - `:tfinal` - Final fine-tuning steps after pruning ends

  ## Example

      config = HfPeftEx.Tuners.Adalora.Config.new(
        total_step: 10000,
        init_r: 12,
        target_r: 8,
        tinit: 200,
        tfinal: 200,
        delta_t: 10,
        target_modules: ["q_proj", "v_proj"]
      )

  ## Math

  The rank budget decreases following a cubic schedule:

      progress = (step - tinit) / (total_step - tfinal - tinit)
      budget = (init_r - target_r) * (1 - progress)^3 + target_r

  Importance score for each singular value:

      I(lambda_i) = |lambda_i * grad(lambda_i)|

  """

  alias HfPeftEx.TaskType

  @type bias :: :none | :all | :lora_only

  @type t :: %__MODULE__{
          # Base config fields
          peft_type: :adalora,
          task_type: TaskType.t() | nil,
          base_model_name_or_path: String.t() | nil,
          revision: String.t() | nil,
          inference_mode: boolean(),
          peft_version: String.t() | nil,
          # LoRA-inherited fields
          lora_alpha: number(),
          lora_dropout: float(),
          fan_in_fan_out: boolean(),
          bias: bias(),
          target_modules: [String.t()] | String.t() | nil,
          exclude_modules: [String.t()] | String.t() | nil,
          modules_to_save: [String.t()] | nil,
          layers_to_transform: [integer()] | integer() | nil,
          layers_pattern: [String.t()] | String.t() | nil,
          # AdaLoRA-specific fields
          init_r: pos_integer(),
          target_r: pos_integer(),
          tinit: non_neg_integer(),
          tfinal: non_neg_integer(),
          delta_t: pos_integer(),
          beta1: float(),
          beta2: float(),
          orth_reg_weight: float(),
          total_step: pos_integer() | nil,
          rank_pattern: map() | nil,
          use_dora: boolean()
        }

  @enforce_keys []
  defstruct peft_type: :adalora,
            task_type: nil,
            base_model_name_or_path: nil,
            revision: nil,
            inference_mode: false,
            peft_version: nil,
            # LoRA-inherited defaults
            lora_alpha: 8,
            lora_dropout: 0.0,
            fan_in_fan_out: false,
            bias: :none,
            target_modules: nil,
            exclude_modules: nil,
            modules_to_save: nil,
            layers_to_transform: nil,
            layers_pattern: nil,
            # AdaLoRA-specific defaults
            init_r: 12,
            target_r: 8,
            tinit: 0,
            tfinal: 0,
            delta_t: 1,
            beta1: 0.85,
            beta2: 0.85,
            orth_reg_weight: 0.5,
            total_step: nil,
            rank_pattern: nil,
            use_dora: false

  @doc """
  Creates a new AdaLoRA configuration.

  ## Options

  ### Required
  - `:total_step` - Total training steps. Must be positive.

  ### AdaLoRA-Specific
  - `:init_r` - Initial rank (default: `12`)
  - `:target_r` - Target rank after pruning (default: `8`)
  - `:tinit` - Warmup steps before pruning (default: `0`)
  - `:tfinal` - Final fine-tuning steps (default: `0`)
  - `:delta_t` - Pruning interval in steps (default: `1`)
  - `:beta1` - EMA coefficient for importance smoothing (default: `0.85`)
  - `:beta2` - EMA coefficient for uncertainty (default: `0.85`)
  - `:orth_reg_weight` - Orthogonal regularization weight (default: `0.5`)

  ### LoRA-Inherited
  - `:lora_alpha` - Scaling factor (default: `8`)
  - `:lora_dropout` - Dropout rate (default: `0.0`)
  - `:target_modules` - Modules to apply AdaLoRA to
  - `:bias` - Bias configuration (`:none`, `:all`, `:lora_only`)

  ## Examples

      # Minimal configuration
      config = Config.new(total_step: 10000)

      # Full configuration
      config = Config.new(
        total_step: 10000,
        init_r: 16,
        target_r: 4,
        tinit: 500,
        tfinal: 500,
        delta_t: 10,
        lora_alpha: 32,
        target_modules: ["q_proj", "v_proj"]
      )

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version(), peft_type: :adalora}
    validate!(config)
  end

  @doc """
  Validates an AdaLoRA configuration.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    validate_total_step!(config)
    validate_schedule!(config)
    validate_ranks!(config)
    validate_dropout!(config)
    validate_dora!(config)
    config
  end

  defp validate_total_step!(%{total_step: nil}) do
    raise ArgumentError, "total_step is required for AdaLoRA"
  end

  defp validate_total_step!(%{total_step: step}) when step <= 0 do
    raise ArgumentError, "total_step must be positive, got: #{step}"
  end

  defp validate_total_step!(_config), do: :ok

  defp validate_schedule!(%{tinit: tinit, tfinal: tfinal, total_step: total}) do
    if tinit >= total - tfinal do
      raise ArgumentError,
            "tinit must be less than total_step - tfinal. " <>
              "Got: tinit=#{tinit}, tfinal=#{tfinal}, total_step=#{total}"
    end
  end

  defp validate_ranks!(%{init_r: init_r, target_r: target_r}) when init_r < target_r do
    raise ArgumentError,
          "init_r must be >= target_r. Got: init_r=#{init_r}, target_r=#{target_r}"
  end

  defp validate_ranks!(_config), do: :ok

  defp validate_dropout!(%{lora_dropout: dropout}) when dropout < 0.0 or dropout >= 1.0 do
    raise ArgumentError, "lora_dropout must be in [0.0, 1.0), got: #{dropout}"
  end

  defp validate_dropout!(_config), do: :ok

  defp validate_dora!(%{use_dora: true}) do
    raise ArgumentError, "AdaLoRA does not support DoRA"
  end

  defp validate_dora!(_config), do: :ok

  @doc """
  Returns the initial scaling factor.

  For AdaLoRA, this uses `lora_alpha / init_r`.
  """
  @spec scaling(t()) :: float()
  def scaling(%__MODULE__{lora_alpha: alpha, init_r: init_r}) do
    alpha / init_r
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
end
