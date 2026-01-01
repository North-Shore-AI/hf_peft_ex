defmodule HfPeftEx.PeftModel do
  @moduledoc """
  Base PEFT model that encompasses various PEFT methods.

  This is the main entry point for creating and managing PEFT models.
  It dispatches to the appropriate tuner (LoRA, etc.) based on the config type.

  ## Example

      # Create a LoRA PEFT model
      config = LoraConfig.new(r: 8, lora_alpha: 16)
      model = PeftModel.new(base_model, config)
      
      # Get trainable parameters
      {trainable, total} = PeftModel.get_nb_trainable_parameters(model)
      
      # Save adapter
      PeftModel.save_pretrained(model, "path/to/adapter")
      
      # Load adapter
      {:ok, model} = PeftModel.from_pretrained(base_model, "path/to/adapter")

  """

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig

  @type t :: %__MODULE__{
          base_model: term(),
          peft_type: atom(),
          adapter_name: String.t(),
          active_adapter: String.t(),
          peft_configs: map(),
          trainable_params: non_neg_integer(),
          total_params: non_neg_integer()
        }

  defstruct [
    :base_model,
    :peft_type,
    adapter_name: "default",
    active_adapter: "default",
    peft_configs: %{},
    trainable_params: 0,
    total_params: 0
  ]

  @doc """
  Creates a new PEFT model from a base model and config.

  ## Arguments

  - `base_model` - The base model to wrap
  - `config` - A PEFT config (e.g., LoraConfig)

  ## Options

  - `:adapter_name` - Name for this adapter (default: "default")

  ## Example

      config = LoraConfig.new(r: 8)
      model = PeftModel.new(base_model, config)

  """
  @spec new(term(), struct(), keyword()) :: t()
  def new(base_model, config, opts \\ [])

  def new(base_model, %LoraConfig{} = config, opts) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    %__MODULE__{
      base_model: base_model,
      peft_type: :lora,
      adapter_name: adapter_name,
      active_adapter: adapter_name,
      peft_configs: %{adapter_name => config},
      trainable_params: 0,
      total_params: 0
    }
  end

  @doc """
  Returns the PEFT configuration.

  If `adapter_name` is provided, returns the config for that adapter.
  Otherwise returns the config for the current active adapter.
  """
  @spec get_peft_config(t(), String.t() | nil) :: struct()
  def get_peft_config(model, adapter_name \\ nil)

  def get_peft_config(%__MODULE__{peft_configs: configs, active_adapter: active}, nil) do
    Map.get(configs, active)
  end

  def get_peft_config(%__MODULE__{peft_configs: configs}, adapter_name) do
    Map.get(configs, adapter_name)
  end

  @doc """
  Returns the number of trainable and total parameters.

  Returns a tuple `{trainable_params, total_params}`.
  """
  @spec get_nb_trainable_parameters(t()) :: {non_neg_integer(), non_neg_integer()}
  def get_nb_trainable_parameters(%__MODULE__{} = model) do
    {model.trainable_params, model.total_params}
  end

  @doc """
  Returns a formatted string with trainable parameter statistics.

  Shows trainable params, total params, and the percentage of trainable params.
  """
  @spec print_trainable_parameters(t()) :: String.t()
  def print_trainable_parameters(%__MODULE__{} = model) do
    {trainable, total} = get_nb_trainable_parameters(model)

    percentage =
      if total > 0 do
        Float.round(trainable / total * 100, 4)
      else
        0.0
      end

    trainable_str = format_number(trainable)
    total_str = format_number(total)

    "trainable params: #{trainable_str} || all params: #{total_str} || trainable%: #{percentage}%"
  end

  defp format_number(num) do
    num
    |> Integer.to_string()
    |> String.graphemes()
    |> Enum.reverse()
    |> Enum.chunk_every(3)
    |> Enum.map_join(",", &Enum.join/1)
    |> String.reverse()
  end

  @doc """
  Saves the PEFT model adapter to a directory.

  Creates an `adapter_config.json` file and optionally saves adapter weights.
  """
  @spec save_pretrained(t(), String.t(), keyword()) :: :ok | {:error, term()}
  def save_pretrained(%__MODULE__{} = model, save_directory, _opts \\ []) do
    config = get_peft_config(model)

    case config do
      %LoraConfig{} = lora_config ->
        LoraConfig.save_pretrained(lora_config, save_directory)

      _ ->
        {:error, :unsupported_peft_type}
    end
  end

  @doc """
  Loads a PEFT model from a pretrained adapter directory.

  ## Arguments

  - `base_model` - The base model to apply the adapter to
  - `model_path` - Path to the saved adapter directory

  ## Example

      {:ok, model} = PeftModel.from_pretrained(base_model, "/path/to/adapter")

  """
  @spec from_pretrained(term(), String.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def from_pretrained(base_model, model_path, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    with {:ok, config} <- LoraConfig.from_pretrained(model_path) do
      model = new(base_model, config, adapter_name: adapter_name)
      {:ok, model}
    end
  end

  @doc """
  Returns a list of active adapter names.
  """
  @spec active_adapters(t()) :: [String.t()]
  def active_adapters(%__MODULE__{active_adapter: active}) do
    [active]
  end

  @doc """
  Sets the active adapter.
  """
  @spec set_adapter(t(), String.t()) :: t()
  def set_adapter(%__MODULE__{} = model, adapter_name) do
    %{model | active_adapter: adapter_name}
  end
end
