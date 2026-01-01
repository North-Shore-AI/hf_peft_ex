defmodule HfPeftEx.Tuners.PromptTuning.Embedding do
  @moduledoc """
  Learnable soft prompt embedding for Prompt Tuning.

  This module implements the prompt embedding layer that holds the learnable
  virtual tokens. These tokens are prepended to input embeddings during the
  forward pass.

  ## Example

      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      embedding = Embedding.new(config)

      # Get embeddings for a batch
      prompt_embeds = Embedding.forward(embedding, batch_size: 4)
      # => Shape: {4, 20, 768}

  ## Initialization Methods

  - `:random` - Random normal initialization with small standard deviation
  - `:sample_vocab` - Sample from vocabulary embeddings
  - `:text` - Initialize from tokenized text (requires tokenizer integration)

  """

  alias HfPeftEx.Tuners.PromptTuning.Config

  @type t :: %__MODULE__{
          embedding: Nx.Tensor.t(),
          num_virtual_tokens: non_neg_integer(),
          token_dim: non_neg_integer()
        }

  defstruct [
    :embedding,
    :num_virtual_tokens,
    :token_dim
  ]

  @doc """
  Creates a new prompt embedding with specified initialization.

  ## Arguments

  - `config` - A `PromptTuning.Config` struct
  - `word_embeddings` - Optional word embeddings tensor for `:sample_vocab` init

  ## Examples

      # Random initialization
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      # Sample from vocabulary
      config = Config.new(num_virtual_tokens: 10, token_dim: 64, prompt_tuning_init: :sample_vocab)
      word_embeddings = model.word_embeddings  # {vocab_size, dim}
      embedding = Embedding.new(config, word_embeddings)

  """
  @spec new(Config.t(), Nx.Tensor.t() | nil) :: t()
  def new(config, word_embeddings \\ nil)

  def new(%Config{} = config, word_embeddings) do
    total_virtual_tokens = Config.total_virtual_tokens(config)
    token_dim = config.token_dim

    embedding_tensor =
      case {config.prompt_tuning_init, word_embeddings, config.inference_mode} do
        # Random initialization (default)
        {:random, _, _} ->
          init_random(total_virtual_tokens, token_dim)

        # Sample from vocabulary embeddings
        {:sample_vocab, word_emb, false} when not is_nil(word_emb) ->
          init_from_vocab(total_virtual_tokens, word_emb)

        # Text initialization would require tokenizer
        {:text, _, false} ->
          raise ArgumentError, "Text initialization requires using from_text/3"

        # Fallback to random for inference mode or missing word embeddings
        _ ->
          init_random(total_virtual_tokens, token_dim)
      end

    %__MODULE__{
      embedding: embedding_tensor,
      num_virtual_tokens: total_virtual_tokens,
      token_dim: token_dim
    }
  end

  # Initialize with random normal values (small std)
  defp init_random(num_tokens, dim) do
    key = Nx.Random.key(System.system_time())
    {tensor, _key} = Nx.Random.normal(key, 0.0, 0.02, shape: {num_tokens, dim})
    tensor
  end

  # Sample random tokens from vocabulary embeddings
  defp init_from_vocab(num_tokens, word_embeddings) do
    {vocab_size, _dim} = Nx.shape(word_embeddings)

    # Generate random indices
    key = Nx.Random.key(System.system_time())
    {indices, _key} = Nx.Random.randint(key, 0, vocab_size, shape: {num_tokens}, type: :s64)

    # Gather embeddings at those indices
    Nx.take(word_embeddings, indices, axis: 0)
  end

  @doc """
  Creates a prompt embedding initialized from text.

  This requires a tokenize function that converts text to embeddings.

  ## Arguments

  - `config` - A `PromptTuning.Config` struct
  - `text` - The initialization text
  - `tokenize_fn` - A function that takes text and returns embeddings

  ## Example

      tokenize_fn = fn text ->
        tokens = Tokenizer.encode(tokenizer, text)
        Nx.take(word_embeddings, tokens, axis: 0)
      end

      embedding = Embedding.from_text(config, "Classify sentiment:", tokenize_fn)

  """
  @spec from_text(Config.t(), String.t(), (String.t() -> Nx.Tensor.t())) :: t()
  def from_text(%Config{} = config, text, tokenize_fn) when is_function(tokenize_fn, 1) do
    total_virtual_tokens = Config.total_virtual_tokens(config)

    # Get embeddings from tokenized text
    text_embeddings = tokenize_fn.(text)
    {actual_tokens, token_dim} = Nx.shape(text_embeddings)

    # Adjust to match num_virtual_tokens
    embedding_tensor =
      cond do
        actual_tokens == total_virtual_tokens ->
          text_embeddings

        actual_tokens > total_virtual_tokens ->
          # Truncate to num_virtual_tokens
          Nx.slice(text_embeddings, [0, 0], [total_virtual_tokens, token_dim])

        actual_tokens < total_virtual_tokens ->
          # Repeat text embeddings to fill
          num_reps = ceil(total_virtual_tokens / actual_tokens)
          repeated = tile_embeddings(text_embeddings, num_reps)
          Nx.slice(repeated, [0, 0], [total_virtual_tokens, token_dim])
      end

    %__MODULE__{
      embedding: embedding_tensor,
      num_virtual_tokens: total_virtual_tokens,
      token_dim: token_dim
    }
  end

  # Tile embeddings along the first axis
  defp tile_embeddings(embeddings, num_reps) do
    embeddings_list = for _ <- 1..num_reps, do: embeddings
    Nx.concatenate(embeddings_list, axis: 0)
  end

  @doc """
  Forward pass: returns prompt embeddings expanded to batch size.

  ## Arguments

  - `prompt_embedding` - The prompt embedding struct
  - `batch_size` - The batch size to expand to

  ## Returns

  A tensor of shape `{batch_size, num_virtual_tokens, token_dim}`.

  ## Example

      output = Embedding.forward(embedding, 4)
      # => Shape: {4, 20, 768}

  """
  @spec forward(t(), pos_integer()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = prompt_embedding, batch_size) do
    # Expand embedding from {num_tokens, dim} to {batch_size, num_tokens, dim}
    # by adding a batch dimension and broadcasting
    prompt_embedding.embedding
    |> Nx.new_axis(0)
    |> Nx.broadcast({batch_size, prompt_embedding.num_virtual_tokens, prompt_embedding.token_dim})
  end

  @doc """
  Returns the trainable parameters as a map.

  ## Example

      params = Embedding.get_trainable_params(embedding)
      # => %{"embedding" => #Nx.Tensor<...>}

  """
  @spec get_trainable_params(t()) :: %{String.t() => Nx.Tensor.t()}
  def get_trainable_params(%__MODULE__{} = prompt_embedding) do
    %{"embedding" => prompt_embedding.embedding}
  end

  @doc """
  Sets the embedding tensor.

  Useful for loading saved embeddings or updating during training.
  """
  @spec set_embedding(t(), Nx.Tensor.t()) :: t()
  def set_embedding(%__MODULE__{} = prompt_embedding, embedding_tensor) do
    %{prompt_embedding | embedding: embedding_tensor}
  end

  @doc """
  Returns the total number of trainable parameters.
  """
  @spec trainable_parameter_count(t()) :: non_neg_integer()
  def trainable_parameter_count(%__MODULE__{} = prompt_embedding) do
    prompt_embedding.num_virtual_tokens * prompt_embedding.token_dim
  end
end
