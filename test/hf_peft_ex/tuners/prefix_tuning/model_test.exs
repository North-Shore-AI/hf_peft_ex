defmodule HfPeftEx.Tuners.PrefixTuning.ModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.{Config, Model}

  describe "new/2" do
    test "creates model with prefix encoder" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)

      assert model.config == config
      assert model.prefix_encoder != nil
      assert model.base_model == base_model
    end

    test "auto-fills config from base model" do
      base_model = create_mock_model()
      config = Config.new(num_virtual_tokens: 10)

      model = Model.new(base_model, config)

      assert model.config.num_layers == 4
      assert model.config.num_attention_heads == 4
      assert model.config.token_dim == 64
    end

    test "uses provided config values over base model" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 6,
          token_dim: 128,
          num_attention_heads: 8
        )

      model = Model.new(base_model, config)

      # Config values should be preserved
      assert model.config.num_layers == 6
      assert model.config.token_dim == 128
      assert model.config.num_attention_heads == 8
    end

    test "creates encoder with projection when configured" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      model = Model.new(base_model, config)

      assert model.prefix_encoder.use_projection == true
      assert model.prefix_encoder.transform != nil
    end
  end

  describe "get_past_key_values/2" do
    test "generates past_key_values for batch" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)
      past_key_values = Model.get_past_key_values(model, 2)

      # Should have entry for each layer
      assert length(past_key_values) == 4

      # Each entry is (key, value) tuple
      {key, value} = hd(past_key_values)
      # Shape: (batch, num_heads, num_tokens, head_dim)
      head_dim = div(64, 4)
      assert Nx.shape(key) == {2, 4, 10, head_dim}
      assert Nx.shape(value) == {2, 4, 10, head_dim}
    end

    test "handles different batch sizes" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2
        )

      model = Model.new(base_model, config)

      for batch_size <- [1, 4, 8] do
        past_key_values = Model.get_past_key_values(model, batch_size)
        {key, _value} = hd(past_key_values)
        assert elem(Nx.shape(key), 0) == batch_size
      end
    end
  end

  describe "get_past_key_values_tensor/2" do
    test "returns full tensor with correct shape" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)
      tensor = Model.get_past_key_values_tensor(model, 2)

      # Shape: (batch, num_layers, 2, num_heads, num_tokens, head_dim)
      head_dim = div(64, 4)
      assert Nx.shape(tensor) == {2, 4, 2, 4, 10, head_dim}
    end
  end

  describe "prepare_attention_mask/3" do
    test "extends attention mask for prefix" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)

      attention_mask = Nx.broadcast(1, {2, 20})
      new_mask = Model.prepare_attention_mask(model, 2, attention_mask)

      # Should be extended by num_virtual_tokens
      assert Nx.shape(new_mask) == {2, 30}
    end

    test "creates full attention mask if none provided" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)

      new_mask = Model.prepare_attention_mask(model, 2, nil)

      assert Nx.shape(new_mask) == {2, 10}
      # All ones
      assert Nx.to_number(Nx.sum(new_mask)) == 20
    end

    test "prepends ones for prefix positions" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)

      # Create mask with some zeros
      attention_mask = Nx.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
      new_mask = Model.prepare_attention_mask(model, 2, attention_mask)

      # Prefix positions should all be 1
      prefix_part = Nx.slice(new_mask, [0, 0], [2, 5])
      assert Nx.to_number(Nx.sum(prefix_part)) == 10

      # Original mask should be preserved after prefix
      original_part = Nx.slice(new_mask, [0, 5], [2, 4])
      assert Nx.equal(original_part, attention_mask) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "get_trainable_params/1" do
    test "returns prefix encoder params" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)
      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "prefix_encoder.embedding")
    end

    test "includes transform params when using projection" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4,
          prefix_projection: true,
          encoder_hidden_size: 128
        )

      model = Model.new(base_model, config)
      params = Model.get_trainable_params(model)

      assert Map.has_key?(params, "prefix_encoder.embedding")
      assert Map.has_key?(params, "prefix_encoder.transform.w1")
      assert Map.has_key?(params, "prefix_encoder.transform.b1")
      assert Map.has_key?(params, "prefix_encoder.transform.w2")
      assert Map.has_key?(params, "prefix_encoder.transform.b2")
    end
  end

  describe "set_trainable_params/2" do
    test "updates prefix encoder params" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 5,
          num_layers: 2,
          token_dim: 32,
          num_attention_heads: 2
        )

      model = Model.new(base_model, config)

      new_embedding = Nx.broadcast(0.5, {5, 2 * 2 * 32})
      new_params = %{"prefix_encoder.embedding" => new_embedding}

      updated_model = Model.set_trainable_params(model, new_params)

      assert Nx.equal(updated_model.prefix_encoder.embedding, new_embedding)
             |> Nx.all()
             |> Nx.to_number() == 1
    end
  end

  describe "param_count/1" do
    test "returns correct count" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)

      # num_virtual_tokens * num_layers * 2 * token_dim
      expected = 10 * 4 * 2 * 64
      assert Model.param_count(model) == expected
    end
  end

  describe "trainable_summary/1" do
    test "returns summary with trainable and total params" do
      base_model = create_mock_model()

      config =
        Config.new(
          num_virtual_tokens: 10,
          num_layers: 4,
          token_dim: 64,
          num_attention_heads: 4
        )

      model = Model.new(base_model, config)
      summary = Model.trainable_summary(model)

      assert summary.trainable_params == 10 * 4 * 2 * 64
      assert summary.prefix_encoder_params == 10 * 4 * 2 * 64
      assert is_map(summary.params_by_layer)
    end
  end

  defp create_mock_model do
    %{
      config: %{
        num_hidden_layers: 4,
        num_attention_heads: 4,
        hidden_size: 64
      }
    }
  end
end
