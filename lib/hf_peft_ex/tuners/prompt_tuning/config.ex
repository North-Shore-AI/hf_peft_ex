defmodule HfPeftEx.Tuners.PromptTuning.Config do
  @moduledoc """
  Configuration class for Prompt Tuning.

  Prompt Tuning adds learnable virtual tokens (soft prompts) to the input embedding.
  These virtual tokens are prepended to the input and learned during training while
  the base model remains frozen.

  ## Key Parameters

  - `:num_virtual_tokens` - Number of soft prompt tokens (default: 20)
  - `:token_dim` - Hidden embedding dimension of the base model
  - `:prompt_tuning_init` - Initialization method (`:random`, `:text`, or `:sample_vocab`)

  ## Example

      config = HfPeftEx.Tuners.PromptTuning.Config.new(
        num_virtual_tokens: 20,
        token_dim: 768,
        prompt_tuning_init: :random
      )

  ## Prompt Tuning Math

  For input embeddings E_1, ..., E_m, prompt tuning prepends P_1, ..., P_n:

      X_combined = [P_1, ..., P_n, E_1, ..., E_m]

  Where P_i are learnable soft prompt embeddings of dimension token_dim.

  """

  alias HfPeftEx.TaskType

  @type init_method :: :random | :text | :sample_vocab

  @type t :: %__MODULE__{
          # Base config fields
          peft_type: :prompt_tuning,
          task_type: TaskType.t() | nil,
          base_model_name_or_path: String.t() | nil,
          revision: String.t() | nil,
          inference_mode: boolean(),
          peft_version: String.t() | nil,
          # Prompt learning config fields
          num_virtual_tokens: pos_integer(),
          token_dim: pos_integer() | nil,
          num_transformer_submodules: pos_integer(),
          num_attention_heads: pos_integer() | nil,
          num_layers: pos_integer() | nil,
          modules_to_save: [String.t()] | nil,
          # Prompt tuning specific fields
          prompt_tuning_init: init_method(),
          prompt_tuning_init_text: String.t() | nil,
          tokenizer_name_or_path: String.t() | nil,
          tokenizer_kwargs: map() | nil
        }

  @enforce_keys []
  defstruct peft_type: :prompt_tuning,
            task_type: nil,
            base_model_name_or_path: nil,
            revision: nil,
            inference_mode: false,
            peft_version: nil,
            # Prompt learning config
            num_virtual_tokens: 20,
            token_dim: nil,
            num_transformer_submodules: 1,
            num_attention_heads: nil,
            num_layers: nil,
            modules_to_save: nil,
            # Prompt tuning specific
            prompt_tuning_init: :random,
            prompt_tuning_init_text: nil,
            tokenizer_name_or_path: nil,
            tokenizer_kwargs: nil

  @doc """
  Creates a new Prompt Tuning configuration.

  ## Options

  ### Prompt Learning Options
  - `:num_virtual_tokens` - Number of soft prompt tokens (default: `20`)
  - `:token_dim` - Hidden embedding dimension of the base model
  - `:num_transformer_submodules` - Number of transformer submodules (default: `1`)

  ### Initialization Options
  - `:prompt_tuning_init` - How to initialize the prompt:
    - `:random` - Random continuous vectors (default)
    - `:sample_vocab` - Sample from vocabulary embeddings
    - `:text` - Initialize from text embeddings
  - `:prompt_tuning_init_text` - Text for initialization (required if init is `:text`)
  - `:tokenizer_name_or_path` - Tokenizer for text init (required if init is `:text`)

  ### General Options
  - `:task_type` - Task type (`:causal_lm`, `:seq_cls`, etc.)
  - `:inference_mode` - Enable inference mode (default: `false`)

  ## Examples

      # Basic configuration with random init
      config = Config.new(num_virtual_tokens: 20, token_dim: 768)

      # Initialize from vocabulary sampling
      config = Config.new(
        num_virtual_tokens: 20,
        token_dim: 768,
        prompt_tuning_init: :sample_vocab
      )

      # Initialize from text
      config = Config.new(
        num_virtual_tokens: 20,
        prompt_tuning_init: :text,
        prompt_tuning_init_text: "Classify if sentiment is positive or negative:",
        tokenizer_name_or_path: "t5-base"
      )

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version(), peft_type: :prompt_tuning}
    validate!(config)
  end

  @doc """
  Validates a Prompt Tuning configuration.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    validate_num_virtual_tokens!(config)
    validate_text_init!(config)
    validate_task_type!(config)
    config
  end

  defp validate_num_virtual_tokens!(%{num_virtual_tokens: n}) when n < 1 do
    raise ArgumentError, "num_virtual_tokens must be at least 1, got: #{n}"
  end

  defp validate_num_virtual_tokens!(_config), do: :ok

  defp validate_text_init!(%{prompt_tuning_init: :text, prompt_tuning_init_text: nil}) do
    raise ArgumentError,
          "prompt_tuning_init_text required when prompt_tuning_init is :text"
  end

  defp validate_text_init!(%{prompt_tuning_init: :text, tokenizer_name_or_path: nil}) do
    raise ArgumentError,
          "tokenizer_name_or_path required when prompt_tuning_init is :text"
  end

  defp validate_text_init!(_config), do: :ok

  defp validate_task_type!(%{task_type: nil}), do: :ok

  defp validate_task_type!(%{task_type: task_type}) do
    unless TaskType.valid?(task_type) do
      raise ArgumentError, "Invalid task_type: #{inspect(task_type)}"
    end
  end

  @doc """
  Returns true if this is a prompt learning method.

  Always returns true for Prompt Tuning.
  """
  @spec prompt_learning?(t()) :: boolean()
  def prompt_learning?(%__MODULE__{}), do: true

  @doc """
  Returns the total number of virtual tokens.

  This is `num_virtual_tokens * num_transformer_submodules`.
  """
  @spec total_virtual_tokens(t()) :: pos_integer()
  def total_virtual_tokens(%__MODULE__{} = config) do
    config.num_virtual_tokens * config.num_transformer_submodules
  end

  @doc """
  Returns the number of trainable parameters.

  Returns `nil` if `token_dim` is not set.
  """
  @spec trainable_params(t()) :: pos_integer() | nil
  def trainable_params(%__MODULE__{token_dim: nil}), do: nil

  def trainable_params(%__MODULE__{} = config) do
    total_virtual_tokens(config) * config.token_dim
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

    # Convert specific fields from uppercase strings to atoms
    map = convert_atom_field(map, :peft_type)
    map = convert_atom_field(map, :task_type)
    map = convert_prompt_tuning_init(map)

    struct(__MODULE__, map)
  end

  defp convert_atom_field(map, key) do
    case Map.get(map, key) do
      nil -> map
      val when is_binary(val) -> Map.put(map, key, val |> String.downcase() |> String.to_atom())
      _ -> map
    end
  end

  defp convert_prompt_tuning_init(map) do
    case Map.get(map, :prompt_tuning_init) do
      nil -> map
      "RANDOM" -> Map.put(map, :prompt_tuning_init, :random)
      "TEXT" -> Map.put(map, :prompt_tuning_init, :text)
      "SAMPLE_VOCAB" -> Map.put(map, :prompt_tuning_init, :sample_vocab)
      val when is_atom(val) -> map
      val when is_binary(val) -> Map.put(map, :prompt_tuning_init, String.to_atom(val))
    end
  end

  @doc """
  Converts the configuration to a JSON string.
  """
  @spec to_json(t()) :: String.t()
  def to_json(%__MODULE__{} = config) do
    config
    |> to_map()
    |> Jason.encode!(pretty: true)
  end

  @doc """
  Creates a configuration from a JSON string.
  """
  @spec from_json(String.t()) :: t()
  def from_json(json) when is_binary(json) do
    json
    |> Jason.decode!()
    |> from_map()
  end

  @doc """
  Saves the Prompt Tuning configuration to a directory.

  Creates an `adapter_config.json` file in the specified directory.
  """
  @spec save_pretrained(t(), String.t()) :: :ok | {:error, term()}
  def save_pretrained(%__MODULE__{} = config, save_directory) do
    File.mkdir_p!(save_directory)
    output_path = Path.join(save_directory, "adapter_config.json")

    json = to_json(config)
    File.write(output_path, json)
  end

  @doc """
  Loads a Prompt Tuning configuration from a directory or file.
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
end
