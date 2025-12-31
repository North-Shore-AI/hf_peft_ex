defmodule HfPeftEx.Config do
  @moduledoc """
  Base configuration module for PEFT adapter models.

  This module defines the common configuration fields shared by all PEFT methods.
  Specific PEFT methods extend this with their own configuration options.

  ## Fields

  - `:peft_type` - The type of PEFT method (e.g., `:lora`, `:ia3`)
  - `:task_type` - The type of task (e.g., `:causal_lm`, `:seq_cls`)
  - `:base_model_name_or_path` - Name or path of the base model
  - `:revision` - Specific model version/revision
  - `:inference_mode` - Whether to use inference mode (frozen weights)
  - `:peft_version` - Version of PEFT used to create the config

  """

  alias HfPeftEx.{PeftType, TaskType}

  @type t :: %__MODULE__{
          peft_type: PeftType.t() | nil,
          task_type: TaskType.t() | nil,
          base_model_name_or_path: String.t() | nil,
          revision: String.t() | nil,
          inference_mode: boolean(),
          peft_version: String.t() | nil,
          auto_mapping: map() | nil
        }

  @enforce_keys []
  defstruct peft_type: nil,
            task_type: nil,
            base_model_name_or_path: nil,
            revision: nil,
            inference_mode: false,
            peft_version: nil,
            auto_mapping: nil

  @doc """
  Creates a new base PEFT configuration.

  ## Options

  - `:peft_type` - The PEFT method type (default: `nil`)
  - `:task_type` - The task type (default: `nil`)
  - `:base_model_name_or_path` - Base model identifier (default: `nil`)
  - `:revision` - Model revision (default: `nil`)
  - `:inference_mode` - Enable inference mode (default: `false`)

  ## Examples

      iex> HfPeftEx.Config.new(peft_type: :lora, task_type: :causal_lm)
      %HfPeftEx.Config{peft_type: :lora, task_type: :causal_lm, ...}

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version()}
    validate!(config)
  end

  @doc """
  Validates a configuration struct.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    if config.peft_type && not PeftType.valid?(config.peft_type) do
      raise ArgumentError, "Invalid peft_type: #{inspect(config.peft_type)}"
    end

    if config.task_type && not TaskType.valid?(config.task_type) do
      raise ArgumentError, "Invalid task_type: #{inspect(config.task_type)}"
    end

    config
  end

  @doc """
  Converts the configuration to a map (for JSON serialization).

  ## Examples

      iex> config = HfPeftEx.Config.new(peft_type: :lora)
      iex> HfPeftEx.Config.to_map(config)
      %{peft_type: :lora, task_type: nil, ...}

  """
  @spec to_map(t()) :: map()
  def to_map(%__MODULE__{} = config) do
    Map.from_struct(config)
  end

  @doc """
  Loads a configuration from a JSON file.

  ## Examples

      iex> HfPeftEx.Config.from_json_file("adapter_config.json")
      {:ok, %HfPeftEx.Config{...}}

  """
  @spec from_json_file(String.t()) :: {:ok, t()} | {:error, term()}
  def from_json_file(path) do
    with {:ok, content} <- File.read(path),
         {:ok, json} <- Jason.decode(content, keys: :atoms) do
      config = from_map(json)
      {:ok, config}
    end
  end

  @doc """
  Creates a configuration from a map.

  This is useful for loading configurations from JSON or other external sources.
  """
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    # Convert string keys to atoms if necessary
    map =
      Map.new(map, fn
        {k, v} when is_binary(k) -> {String.to_existing_atom(k), v}
        pair -> pair
      end)

    # Convert peft_type string to atom
    map =
      case Map.get(map, :peft_type) do
        nil ->
          map

        type when is_binary(type) ->
          Map.put(map, :peft_type, type |> String.downcase() |> String.to_atom())

        type when is_atom(type) ->
          map
      end

    # Convert task_type string to atom
    map =
      case Map.get(map, :task_type) do
        nil ->
          map

        type when is_binary(type) ->
          Map.put(map, :task_type, type |> String.downcase() |> String.to_atom())

        type when is_atom(type) ->
          map
      end

    struct(__MODULE__, map)
  end

  @doc """
  Saves the configuration to a JSON file.

  ## Examples

      iex> config = HfPeftEx.Config.new(peft_type: :lora)
      iex> HfPeftEx.Config.save_pretrained(config, "./my-adapter")
      :ok

  """
  @spec save_pretrained(t(), String.t()) :: :ok | {:error, term()}
  def save_pretrained(%__MODULE__{} = config, save_directory) do
    File.mkdir_p!(save_directory)
    output_path = Path.join(save_directory, "adapter_config.json")

    json =
      config
      |> to_map()
      |> convert_atoms_to_strings()
      |> Jason.encode!(pretty: true)

    File.write(output_path, json)
  end

  # Convert atom values to uppercase strings for JSON compatibility
  defp convert_atoms_to_strings(map) do
    Map.new(map, fn
      {k, v} when is_atom(v) and not is_nil(v) and not is_boolean(v) ->
        {k, v |> Atom.to_string() |> String.upcase()}

      pair ->
        pair
    end)
  end

  @doc """
  Returns true if this configuration is for a prompt learning method.

  Prompt learning methods include prompt tuning, prefix tuning, P-tuning, etc.
  """
  @spec prompt_learning?(t()) :: boolean()
  def prompt_learning?(%__MODULE__{peft_type: peft_type}) do
    PeftType.prompt_learning?(peft_type)
  end
end
