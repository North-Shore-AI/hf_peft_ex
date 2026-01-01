defmodule HfPeftEx.Tuners.IA3.Config do
  @moduledoc """
  Configuration class for IA3 (Infused Adapter by Inhibiting and Amplifying Activations).

  IA3 learns multiplicative scaling vectors instead of low-rank matrices. It's simpler
  than LoRA and has minimal parameter overhead - only one scaling vector per layer.

  ## Key Parameters

  - `:target_modules` - Which modules to apply IA3 to
  - `:feedforward_modules` - Modules treated as feedforward (scaling applied to input)
  - `:init_ia3_weights` - Initialize vectors to ones (identity) or random

  ## Example

      config = HfPeftEx.Tuners.IA3.Config.new(
        target_modules: ["q_proj", "v_proj", "down_proj"],
        feedforward_modules: ["down_proj"],
        init_ia3_weights: true
      )

  ## IA3 Math

  For a pretrained weight matrix W, IA3 scales activations by a learned vector l:

      y = (Wx) * l  (for non-feedforward layers)
      y = W(x * l)  (for feedforward layers)

  The only trainable parameters are the scaling vectors l.

  """

  alias HfPeftEx.{PeftType, TaskType}

  @type t :: %__MODULE__{
          # Base config fields
          peft_type: PeftType.t(),
          task_type: TaskType.t() | nil,
          base_model_name_or_path: String.t() | nil,
          revision: String.t() | nil,
          inference_mode: boolean(),
          peft_version: String.t() | nil,
          # IA3-specific fields
          target_modules: [String.t()] | String.t() | nil,
          feedforward_modules: [String.t()] | String.t() | nil,
          exclude_modules: [String.t()] | String.t() | nil,
          fan_in_fan_out: boolean(),
          modules_to_save: [String.t()] | nil,
          init_ia3_weights: boolean()
        }

  @enforce_keys []
  defstruct peft_type: :ia3,
            task_type: nil,
            base_model_name_or_path: nil,
            revision: nil,
            inference_mode: false,
            peft_version: nil,
            # IA3 defaults
            target_modules: nil,
            feedforward_modules: nil,
            exclude_modules: nil,
            fan_in_fan_out: false,
            modules_to_save: nil,
            init_ia3_weights: true

  @doc """
  Creates a new IA3 configuration.

  ## Options

  ### Target Modules
  - `:target_modules` - Module names to apply IA3. Can be:
    - List of strings: `["q_proj", "v_proj"]`
    - Regex string: `".*_proj"`
    - `nil` to use model defaults
  - `:feedforward_modules` - Modules where scaling is applied to input instead of output.
    Must be a subset of `target_modules`.
  - `:exclude_modules` - Modules to exclude from IA3

  ### Initialization
  - `:init_ia3_weights` - Initialize vectors to ones (default: `true`).
    When `true`, the layer starts as an identity operation.
  - `:fan_in_fan_out` - Set `true` if weight is stored as (fan_in, fan_out)

  ### Training Configuration
  - `:task_type` - Task type (`:causal_lm`, `:seq_cls`, etc.)
  - `:modules_to_save` - Extra modules to train and save

  ## Examples

      # Basic configuration
      config = HfPeftEx.Tuners.IA3.Config.new(
        target_modules: ["q_proj", "v_proj"]
      )

      # With feedforward modules
      config = HfPeftEx.Tuners.IA3.Config.new(
        target_modules: ["q_proj", "v_proj", "down_proj"],
        feedforward_modules: ["down_proj"]
      )

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version(), peft_type: :ia3}
    validate!(config)
  end

  @doc """
  Validates an IA3 configuration.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    validate_feedforward_subset!(config)
    validate_task_type!(config)
    config
  end

  defp validate_feedforward_subset!(%{
         feedforward_modules: ff_modules,
         target_modules: target_modules
       })
       when is_list(ff_modules) and is_list(target_modules) do
    ff_set = MapSet.new(ff_modules)
    target_set = MapSet.new(target_modules)

    unless MapSet.subset?(ff_set, target_set) do
      raise ArgumentError,
            "feedforward_modules must be a subset of target_modules. " <>
              "Got feedforward_modules: #{inspect(ff_modules)}, " <>
              "target_modules: #{inspect(target_modules)}"
    end
  end

  defp validate_feedforward_subset!(_config), do: :ok

  defp validate_task_type!(%{task_type: nil}), do: :ok

  defp validate_task_type!(%{task_type: task_type}) do
    unless TaskType.valid?(task_type) do
      raise ArgumentError, "Invalid task_type: #{inspect(task_type)}"
    end
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

    struct(__MODULE__, map)
  end

  defp convert_atom_field(map, key) do
    case Map.get(map, key) do
      nil -> map
      val when is_binary(val) -> Map.put(map, key, val |> String.downcase() |> String.to_atom())
      _ -> map
    end
  end

  @doc """
  Saves the IA3 configuration to a directory.

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
  Loads an IA3 configuration from a directory or file.
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
  Returns the trainable parameter count for an IA3 layer.

  IA3 adds only `d` parameters per layer, where `d` is:
  - `in_features` for feedforward layers
  - `out_features` for non-feedforward layers
  """
  @spec trainable_params(t(), pos_integer(), pos_integer(), boolean()) :: pos_integer()
  def trainable_params(%__MODULE__{}, in_features, out_features, is_feedforward) do
    if is_feedforward do
      in_features
    else
      out_features
    end
  end
end
