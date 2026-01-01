defmodule HfPeftEx.Tuners.PrefixTuning.Config do
  @moduledoc """
  Configuration class for Prefix Tuning.

  Prefix Tuning prepends trainable prefix tokens to the keys and values of each
  attention layer. Unlike Prompt Tuning which only modifies the input embeddings,
  Prefix Tuning modifies all transformer layers.

  ## Key Parameters

  - `:num_virtual_tokens` - Number of prefix tokens per layer (default: 20)
  - `:token_dim` - Hidden dimension of the base model
  - `:num_layers` - Number of transformer layers in the base model
  - `:num_attention_heads` - Number of attention heads per layer
  - `:prefix_projection` - Whether to use MLP projection for reparameterization (default: false)
  - `:encoder_hidden_size` - Hidden size of the MLP encoder (required if prefix_projection is true)

  ## Example

      config = HfPeftEx.Tuners.PrefixTuning.Config.new(
        num_virtual_tokens: 20,
        num_layers: 12,
        token_dim: 768,
        num_attention_heads: 12,
        prefix_projection: true,
        encoder_hidden_size: 512
      )

  ## Prefix Tuning Math

  For each attention layer l, prefix tuning prepends prefix K,V pairs:

      K_l = [P_k^l, K_input]
      V_l = [P_v^l, V_input]
      Attention_l = softmax(Q @ K_l^T / sqrt(d)) @ V_l

  With projection (reparameterization):

      P = Tanh(E @ W1 + b1) @ W2 + b2
      P_reshaped = reshape(P, (num_layers, 2, num_heads, num_tokens, head_dim))

  """

  alias HfPeftEx.TaskType

  @type t :: %__MODULE__{
          # Base config fields
          peft_type: :prefix_tuning,
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
          # Prefix tuning specific fields
          prefix_projection: boolean(),
          encoder_hidden_size: pos_integer() | nil
        }

  @enforce_keys []
  defstruct peft_type: :prefix_tuning,
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
            # Prefix tuning specific
            prefix_projection: false,
            encoder_hidden_size: nil

  @doc """
  Creates a new Prefix Tuning configuration.

  ## Options

  ### Prompt Learning Options
  - `:num_virtual_tokens` - Number of prefix tokens per layer (default: `20`)
  - `:token_dim` - Hidden embedding dimension of the base model
  - `:num_layers` - Number of transformer layers
  - `:num_attention_heads` - Number of attention heads per layer
  - `:num_transformer_submodules` - Number of transformer submodules (default: `1`)

  ### Prefix Tuning Options
  - `:prefix_projection` - Whether to use MLP projection (default: `false`)
  - `:encoder_hidden_size` - Hidden size of the MLP encoder (required if prefix_projection is true)

  ### General Options
  - `:task_type` - Task type (`:causal_lm`, `:seq_cls`, etc.)
  - `:inference_mode` - Enable inference mode (default: `false`)

  ## Examples

      # Basic configuration without projection
      config = Config.new(
        num_virtual_tokens: 20,
        num_layers: 12,
        token_dim: 768,
        num_attention_heads: 12
      )

      # With MLP projection (reparameterization)
      config = Config.new(
        num_virtual_tokens: 20,
        num_layers: 12,
        token_dim: 768,
        num_attention_heads: 12,
        prefix_projection: true,
        encoder_hidden_size: 512
      )

  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    config = struct(__MODULE__, opts)
    config = %{config | peft_version: HfPeftEx.version(), peft_type: :prefix_tuning}
    validate!(config)
  end

  @doc """
  Validates a Prefix Tuning configuration.

  Raises `ArgumentError` if the configuration is invalid.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = config) do
    validate_num_virtual_tokens!(config)
    validate_prefix_projection!(config)
    validate_encoder_hidden_size!(config)
    validate_task_type!(config)
    config
  end

  defp validate_num_virtual_tokens!(%{num_virtual_tokens: n}) when n < 1 do
    raise ArgumentError, "num_virtual_tokens must be at least 1, got: #{n}"
  end

  defp validate_num_virtual_tokens!(_config), do: :ok

  defp validate_prefix_projection!(%{prefix_projection: true, encoder_hidden_size: nil}) do
    raise ArgumentError,
          "encoder_hidden_size required when prefix_projection is true"
  end

  defp validate_prefix_projection!(_config), do: :ok

  defp validate_encoder_hidden_size!(%{
         prefix_projection: true,
         encoder_hidden_size: size
       })
       when is_integer(size) and size < 1 do
    raise ArgumentError, "encoder_hidden_size must be positive, got: #{size}"
  end

  defp validate_encoder_hidden_size!(_config), do: :ok

  defp validate_task_type!(%{task_type: nil}), do: :ok

  defp validate_task_type!(%{task_type: task_type}) do
    unless TaskType.valid?(task_type) do
      raise ArgumentError, "Invalid task_type: #{inspect(task_type)}"
    end
  end

  @doc """
  Returns true if this is a prompt learning method.

  Always returns true for Prefix Tuning.
  """
  @spec prompt_learning?(t()) :: boolean()
  def prompt_learning?(%__MODULE__{}), do: true

  @doc """
  Calculates the total prefix embedding dimension.

  This is `num_layers * 2 * token_dim` (2 for key and value per layer).

  Returns `nil` if `num_layers` or `token_dim` is not set.
  """
  @spec prefix_embedding_dim(t()) :: pos_integer() | nil
  def prefix_embedding_dim(%__MODULE__{num_layers: nil}), do: nil
  def prefix_embedding_dim(%__MODULE__{token_dim: nil}), do: nil

  def prefix_embedding_dim(%__MODULE__{num_layers: num_layers, token_dim: token_dim}) do
    num_layers * 2 * token_dim
  end

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

  Without projection:
    `num_virtual_tokens * num_layers * 2 * token_dim`

  With projection:
    embedding params + MLP params

  Returns `nil` if required dimensions are not set.
  """
  @spec trainable_params(t()) :: pos_integer() | nil
  def trainable_params(%__MODULE__{num_layers: nil}), do: nil
  def trainable_params(%__MODULE__{token_dim: nil}), do: nil

  def trainable_params(%__MODULE__{prefix_projection: false} = config) do
    config.num_virtual_tokens * config.num_layers * 2 * config.token_dim
  end

  def trainable_params(%__MODULE__{prefix_projection: true, encoder_hidden_size: nil}), do: nil

  def trainable_params(%__MODULE__{prefix_projection: true} = config) do
    output_dim = config.num_layers * 2 * config.token_dim

    # Embedding: num_virtual_tokens * token_dim
    embedding_params = config.num_virtual_tokens * config.token_dim

    # MLP layer 1: token_dim * encoder_hidden_size + encoder_hidden_size (bias)
    layer1_params = config.token_dim * config.encoder_hidden_size + config.encoder_hidden_size

    # MLP layer 2: encoder_hidden_size * output_dim + output_dim (bias)
    layer2_params = config.encoder_hidden_size * output_dim + output_dim

    embedding_params + layer1_params + layer2_params
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
  Saves the Prefix Tuning configuration to a directory.

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
  Loads a Prefix Tuning configuration from a directory or file.
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
