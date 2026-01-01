defmodule HfPeftEx.Tuners.PromptTuning.EmbeddingTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.{Config, Embedding}

  describe "new/2 with random init" do
    test "creates embedding with correct shape" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      assert Nx.shape(embedding.embedding) == {10, 64}
      assert embedding.num_virtual_tokens == 10
      assert embedding.token_dim == 64
    end

    test "initializes with small random values" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      mean = Nx.mean(embedding.embedding) |> Nx.to_number()
      std = Nx.standard_deviation(embedding.embedding) |> Nx.to_number()

      # Random normal with small std, mean should be close to 0
      assert abs(mean) < 0.5
      # Std should be around 0.02 (default init)
      assert std < 0.5
    end

    test "creates embedding when word_embeddings is nil" do
      config = Config.new(num_virtual_tokens: 5, token_dim: 32)
      embedding = Embedding.new(config, nil)

      assert Nx.shape(embedding.embedding) == {5, 32}
    end
  end

  describe "new/2 with num_transformer_submodules" do
    test "creates embedding for total virtual tokens" do
      config = Config.new(num_virtual_tokens: 5, token_dim: 64, num_transformer_submodules: 2)
      embedding = Embedding.new(config)

      # Total = num_virtual_tokens * num_transformer_submodules = 5 * 2 = 10
      assert Nx.shape(embedding.embedding) == {10, 64}
      assert embedding.num_virtual_tokens == 10
      assert embedding.token_dim == 64
    end
  end

  describe "new/2 with sample_vocab init" do
    test "samples from word embeddings" do
      config =
        Config.new(
          num_virtual_tokens: 5,
          token_dim: 64,
          prompt_tuning_init: :sample_vocab
        )

      # Mock vocabulary embeddings (vocab_size=1000, dim=64)
      key = Nx.Random.key(42)
      {word_embeddings, _key} = Nx.Random.uniform(key, shape: {1000, 64})
      embedding = Embedding.new(config, word_embeddings)

      assert Nx.shape(embedding.embedding) == {5, 64}
    end

    test "embedding values come from word embeddings" do
      config =
        Config.new(
          num_virtual_tokens: 3,
          token_dim: 4,
          prompt_tuning_init: :sample_vocab
        )

      # Create distinct word embeddings
      word_embeddings =
        Nx.tensor([
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
        ])

      embedding = Embedding.new(config, word_embeddings)

      # Each row should be one of the word embeddings (a one-hot vector)
      # Check that each row sums to 1 (since they're one-hot)
      row_sums = Nx.sum(embedding.embedding, axes: [1])

      Enum.each(0..2, fn i ->
        sum = Nx.to_number(row_sums[i])
        assert_in_delta sum, 1.0, 0.001
      end)
    end
  end

  describe "forward/2" do
    test "returns batch of prompt embeddings" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 4)

      assert Nx.shape(output) == {4, 10, 64}
    end

    test "all batch items are identical" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 4)

      first = output[0]
      second = output[1]
      third = output[2]

      assert Nx.all_close(first, second) |> Nx.to_number() == 1
      assert Nx.all_close(first, third) |> Nx.to_number() == 1
    end

    test "batch size 1 returns correct shape" do
      config = Config.new(num_virtual_tokens: 5, token_dim: 32)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 1)

      assert Nx.shape(output) == {1, 5, 32}
    end

    test "large batch size works" do
      config = Config.new(num_virtual_tokens: 20, token_dim: 128)
      embedding = Embedding.new(config)

      output = Embedding.forward(embedding, 32)

      assert Nx.shape(output) == {32, 20, 128}
    end
  end

  describe "get_trainable_params/1" do
    test "returns embedding tensor" do
      config = Config.new(num_virtual_tokens: 10, token_dim: 64)
      embedding = Embedding.new(config)

      params = Embedding.get_trainable_params(embedding)

      assert Map.has_key?(params, "embedding")
      assert Nx.shape(params["embedding"]) == {10, 64}
    end

    test "returned embedding is the same as stored" do
      config = Config.new(num_virtual_tokens: 5, token_dim: 32)
      embedding = Embedding.new(config)

      params = Embedding.get_trainable_params(embedding)

      assert Nx.all_close(params["embedding"], embedding.embedding) |> Nx.to_number() == 1
    end
  end

  describe "set_embedding/2" do
    test "replaces embedding tensor" do
      config = Config.new(num_virtual_tokens: 5, token_dim: 4)
      embedding = Embedding.new(config)

      new_embedding_tensor = Nx.broadcast(1.0, {5, 4})
      updated = Embedding.set_embedding(embedding, new_embedding_tensor)

      assert Nx.all_close(updated.embedding, new_embedding_tensor) |> Nx.to_number() == 1
    end
  end

  describe "trainable_parameter_count/1" do
    test "returns correct count" do
      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      embedding = Embedding.new(config)

      count = Embedding.trainable_parameter_count(embedding)

      # 20 * 768 = 15,360
      assert count == 15_360
    end
  end
end
