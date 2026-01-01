defmodule HfPeftEx.Tuners.PromptTuning.Model do
  @moduledoc """
  Prompt Tuning model wrapper.

  This module provides the main interface for using Prompt Tuning with a base model.
  It handles:
  - Creating and managing the prompt encoder
  - Preparing inputs by prepending soft prompts
  - Saving and loading prompt tuning adapters

  ## Example

      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      model = Model.new(base_model, config)

      # Prepare inputs for the base model
      {combined_embeds, attention_mask} = Model.prepare_inputs(
        model,
        input_embeds,
        attention_mask: mask
      )

      # Get trainable parameters for optimization
      params = Model.get_trainable_params(model)

  ## Forward Flow

  1. Get input embeddings from base model
  2. Get soft prompt embeddings from prompt encoder
  3. Concatenate: [prompts, inputs]
  4. Update attention mask
  5. Forward through base model

  """

  alias HfPeftEx.Tuners.PromptTuning.{Config, Embedding}

  @type t :: %__MODULE__{
          base_model: term(),
          config: Config.t(),
          prompt_encoder: Embedding.t(),
          word_embeddings: Nx.Tensor.t() | nil,
          adapter_name: String.t()
        }

  defstruct [
    :base_model,
    :config,
    :prompt_encoder,
    :word_embeddings,
    adapter_name: "default"
  ]

  @doc """
  Creates a new Prompt Tuning model wrapper.

  ## Arguments

  - `base_model` - The base model to wrap (should have `:embeddings` key)
  - `config` - A `PromptTuning.Config` struct

  ## Options

  - `:adapter_name` - Name for this adapter (default: "default")

  ## Example

      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      model = Model.new(base_model, config)

  """
  @spec new(term(), Config.t(), keyword()) :: t()
  def new(base_model, %Config{} = config, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    # Extract word embeddings from base model
    word_embeddings = get_word_embeddings(base_model)

    # Create prompt encoder with appropriate initialization
    prompt_encoder = Embedding.new(config, word_embeddings)

    %__MODULE__{
      base_model: base_model,
      config: config,
      prompt_encoder: prompt_encoder,
      word_embeddings: word_embeddings,
      adapter_name: adapter_name
    }
  end

  # Extracts word embeddings from the base model
  defp get_word_embeddings(%{embeddings: embeddings}), do: embeddings
  defp get_word_embeddings(%{embed_tokens: embed_tokens}), do: embed_tokens
  defp get_word_embeddings(_), do: nil

  @doc """
  Prepares inputs by prepending soft prompts to input embeddings.

  Returns a tuple of `{combined_embeddings, attention_mask}`.

  ## Arguments

  - `model` - The Prompt Tuning model
  - `input_embeds` - Input embeddings tensor of shape `{batch, seq_len, dim}`

  ## Options

  - `:attention_mask` - Optional attention mask of shape `{batch, seq_len}`

  ## Returns

  - `{combined_embeds, new_attention_mask}` where:
    - `combined_embeds` has shape `{batch, num_virtual_tokens + seq_len, dim}`
    - `new_attention_mask` has shape `{batch, num_virtual_tokens + seq_len}`

  ## Example

      {combined, mask} = Model.prepare_inputs(model, input_embeds, attention_mask: attention_mask)

  """
  @spec prepare_inputs(t(), Nx.Tensor.t(), keyword()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def prepare_inputs(%__MODULE__{} = model, input_embeds, opts \\ []) do
    attention_mask = Keyword.get(opts, :attention_mask)

    {batch_size, seq_len, _dim} = Nx.shape(input_embeds)
    num_virtual_tokens = model.config |> Config.total_virtual_tokens()

    # Get prompt embeddings for this batch
    prompt_embeds = Embedding.forward(model.prompt_encoder, batch_size)

    # Concatenate: [prompts, inputs]
    combined_embeds = Nx.concatenate([prompt_embeds, input_embeds], axis: 1)

    # Create or update attention mask
    new_mask =
      case attention_mask do
        nil ->
          # Create all-ones mask for full sequence
          Nx.broadcast(1, {batch_size, num_virtual_tokens + seq_len})

        mask ->
          # Prepend ones for prompt tokens
          prefix_mask = Nx.broadcast(1, {batch_size, num_virtual_tokens})
          Nx.concatenate([prefix_mask, mask], axis: 1)
      end

    {combined_embeds, new_mask}
  end

  @doc """
  Returns the trainable parameters for this model.

  Only the prompt embedding parameters are trainable.

  ## Example

      params = Model.get_trainable_params(model)
      # => %{"prompt_encoder.embedding" => #Nx.Tensor<...>}

  """
  @spec get_trainable_params(t()) :: %{String.t() => Nx.Tensor.t()}
  def get_trainable_params(%__MODULE__{} = model) do
    %{"prompt_encoder.embedding" => model.prompt_encoder.embedding}
  end

  @doc """
  Returns the total number of trainable parameters.
  """
  @spec trainable_parameter_count(t()) :: non_neg_integer()
  def trainable_parameter_count(%__MODULE__{} = model) do
    Embedding.trainable_parameter_count(model.prompt_encoder)
  end

  @doc """
  Sets the prompt embedding tensor.

  Useful for loading saved embeddings or updating during training.
  """
  @spec set_prompt_embedding(t(), Nx.Tensor.t()) :: t()
  def set_prompt_embedding(%__MODULE__{} = model, embedding_tensor) do
    updated_encoder = Embedding.set_embedding(model.prompt_encoder, embedding_tensor)
    %{model | prompt_encoder: updated_encoder}
  end

  @doc """
  Returns true if this is a prompt learning model.

  Always returns true for Prompt Tuning.
  """
  @spec prompt_learning?(t()) :: boolean()
  def prompt_learning?(%__MODULE__{}), do: true

  @doc """
  Returns the PEFT configuration for this model.
  """
  @spec get_peft_config(t()) :: Config.t()
  def get_peft_config(%__MODULE__{config: config}), do: config

  @doc """
  Saves the Prompt Tuning model to a directory.

  Creates:
  - `adapter_config.json` - Configuration file
  - `adapter_model.bin` - Prompt embedding weights

  """
  @spec save_pretrained(t(), String.t()) :: :ok | {:error, term()}
  def save_pretrained(%__MODULE__{} = model, save_directory) do
    File.mkdir_p!(save_directory)

    # Save config
    :ok = Config.save_pretrained(model.config, save_directory)

    # Save prompt embeddings
    weights_path = Path.join(save_directory, "adapter_model.bin")
    save_embedding_weights(model.prompt_encoder.embedding, weights_path)
  end

  # Saves embedding weights to a binary file using Nx.serialize
  defp save_embedding_weights(embedding, path) do
    binary = Nx.serialize(embedding)
    File.write!(path, binary)
    :ok
  end

  @doc """
  Loads a Prompt Tuning model from a directory.

  ## Arguments

  - `base_model` - The base model to wrap
  - `path` - Path to the saved adapter directory

  ## Returns

  `{:ok, model}` or `{:error, reason}`

  """
  @spec from_pretrained(term(), String.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def from_pretrained(base_model, path, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    with {:ok, config} <- Config.from_pretrained(path),
         {:ok, embedding} <- load_embedding_weights(path) do
      model = new(base_model, config, adapter_name: adapter_name)
      model = set_prompt_embedding(model, embedding)
      {:ok, model}
    end
  end

  # Loads embedding weights from a binary file
  defp load_embedding_weights(dir) do
    weights_path = Path.join(dir, "adapter_model.bin")

    case File.read(weights_path) do
      {:ok, binary} ->
        embedding = Nx.deserialize(binary)
        {:ok, embedding}

      {:error, reason} ->
        {:error, {:file_read_error, reason}}
    end
  end
end
