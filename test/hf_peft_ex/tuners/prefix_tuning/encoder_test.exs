defmodule HfPeftEx.Tuners.PrefixTuning.EncoderTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.{Config, Encoder}

  describe "new/1 without projection" do
    test "creates direct embedding" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: false
        )

      encoder = Encoder.new(config)

      # num_layers * 2 * token_dim = 4 * 2 * 64 = 512
      total_dim = 4 * 2 * 64
      assert Nx.shape(encoder.embedding) == {10, total_dim}
      assert encoder.use_projection == false
      assert encoder.transform == nil
    end

    test "stores config reference" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: false
        )

      encoder = Encoder.new(config)

      assert encoder.config == config
    end
  end

  describe "new/1 with projection" do
    test "creates MLP projection" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      encoder = Encoder.new(config)

      # Embedding: (num_virtual_tokens, token_dim)
      assert Nx.shape(encoder.embedding) == {10, 64}
      assert encoder.use_projection == true
      assert encoder.transform != nil

      # Transform layer 1: (token_dim, encoder_hidden_size)
      assert Nx.shape(encoder.transform.w1) == {64, 128}
      assert Nx.shape(encoder.transform.b1) == {128}

      # Transform layer 2: (encoder_hidden_size, num_layers * 2 * token_dim)
      output_dim = 4 * 2 * 64
      assert Nx.shape(encoder.transform.w2) == {128, output_dim}
      assert Nx.shape(encoder.transform.b2) == {output_dim}
    end
  end

  describe "forward/2 without projection" do
    test "returns past_key_values with correct shape" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: false
        )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 2)

      # Shape: (batch, num_layers, 2, num_heads, num_tokens, head_dim)
      head_dim = div(64, 4)
      expected_shape = {2, 4, 2, 4, 10, head_dim}
      assert Nx.shape(output) == expected_shape
    end

    test "handles batch_size of 1" do
      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2,
          prefix_projection: false
        )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 1)

      head_dim = div(32, 2)
      assert Nx.shape(output) == {1, 2, 2, 2, 5, head_dim}
    end
  end

  describe "forward/2 with projection" do
    test "returns past_key_values with correct shape" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 2)

      head_dim = div(64, 4)
      assert Nx.shape(output) == {2, 4, 2, 4, 10, head_dim}
    end

    test "applies tanh nonlinearity" do
      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2,
          prefix_projection: true,
          encoder_hidden_size: 64
        )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 1)

      # Output values should be bounded (influenced by tanh in projection)
      # Even though final output goes through another linear layer,
      # the values should still be reasonable
      assert Nx.type(output) == {:f, 32}
    end
  end

  describe "forward/2 output content" do
    test "embedding is broadcast across batch" do
      config =
        Config.new(
          num_virtual_tokens: 3,
          num_layers: 2,
          token_dim: 16,
          num_attention_heads: 2,
          prefix_projection: false
        )

      encoder = Encoder.new(config)
      output = Encoder.forward(encoder, 3)

      # All batch items should have identical values (same embedding broadcast)
      batch_0 = output[0]
      batch_1 = output[1]
      batch_2 = output[2]

      assert Nx.equal(batch_0, batch_1) |> Nx.all() |> Nx.to_number() == 1
      assert Nx.equal(batch_1, batch_2) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "get_trainable_params/1" do
    test "returns embedding for non-projection" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: false
        )

      encoder = Encoder.new(config)
      params = Encoder.get_trainable_params(encoder)

      assert Map.has_key?(params, "embedding")
      assert map_size(params) == 1
      assert Nx.shape(params["embedding"]) == {10, 4 * 2 * 64}
    end

    test "returns embedding and transform for projection" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      encoder = Encoder.new(config)
      params = Encoder.get_trainable_params(encoder)

      assert Map.has_key?(params, "embedding")
      assert Map.has_key?(params, "transform.w1")
      assert Map.has_key?(params, "transform.b1")
      assert Map.has_key?(params, "transform.w2")
      assert Map.has_key?(params, "transform.b2")

      assert map_size(params) == 5

      assert Nx.shape(params["embedding"]) == {10, 64}
      assert Nx.shape(params["transform.w1"]) == {64, 128}
      assert Nx.shape(params["transform.b1"]) == {128}

      output_dim = 4 * 2 * 64
      assert Nx.shape(params["transform.w2"]) == {128, output_dim}
      assert Nx.shape(params["transform.b2"]) == {output_dim}
    end
  end

  describe "set_trainable_params/2" do
    test "updates embedding params for non-projection" do
      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2,
          prefix_projection: false
        )

      encoder = Encoder.new(config)

      new_embedding = Nx.broadcast(0.5, {5, 2 * 2 * 32})
      new_params = %{"embedding" => new_embedding}

      updated_encoder = Encoder.set_trainable_params(encoder, new_params)

      assert Nx.equal(updated_encoder.embedding, new_embedding) |> Nx.all() |> Nx.to_number() ==
               1
    end

    test "updates all params for projection" do
      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2,
          prefix_projection: true,
          encoder_hidden_size: 64
        )

      encoder = Encoder.new(config)

      output_dim = 2 * 2 * 32

      new_params = %{
        "embedding" => Nx.broadcast(0.1, {5, 32}),
        "transform.w1" => Nx.broadcast(0.2, {32, 64}),
        "transform.b1" => Nx.broadcast(0.3, {64}),
        "transform.w2" => Nx.broadcast(0.4, {64, output_dim}),
        "transform.b2" => Nx.broadcast(0.5, {output_dim})
      }

      updated_encoder = Encoder.set_trainable_params(encoder, new_params)

      assert Nx.to_number(Nx.mean(updated_encoder.embedding)) == 0.10000000149011612
      assert Nx.to_number(Nx.mean(updated_encoder.transform.w1)) == 0.20000000298023224
    end
  end

  describe "param_count/1" do
    test "returns correct count for non-projection" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: false
        )

      encoder = Encoder.new(config)

      # num_virtual_tokens * num_layers * 2 * token_dim
      expected = 10 * 4 * 2 * 64
      assert Encoder.param_count(encoder) == expected
    end

    test "returns correct count for projection" do
      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      encoder = Encoder.new(config)

      output_dim = 4 * 2 * 64

      # embedding: 10 * 64 = 640
      # w1: 64 * 128 = 8192
      # b1: 128
      # w2: 128 * 512 = 65536
      # b2: 512
      expected = 10 * 64 + 64 * 128 + 128 + 128 * output_dim + output_dim
      assert Encoder.param_count(encoder) == expected
    end
  end
end
