defmodule HfPeftEx.Tuners.PrefixTuning.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PrefixTuning.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :prefix_tuning
      assert config.num_virtual_tokens == 20
      assert config.prefix_projection == false
      assert config.encoder_hidden_size == nil
      assert config.num_transformer_submodules == 1
    end

    test "accepts custom num_virtual_tokens" do
      config = Config.new(num_virtual_tokens: 30)
      assert config.num_virtual_tokens == 30
    end

    test "accepts prefix_projection without encoder_hidden_size when false" do
      config = Config.new(prefix_projection: false)
      assert config.prefix_projection == false
      assert config.encoder_hidden_size == nil
    end

    test "accepts prefix_projection with encoder_hidden_size" do
      config = Config.new(prefix_projection: true, encoder_hidden_size: 512)
      assert config.prefix_projection == true
      assert config.encoder_hidden_size == 512
    end

    test "validates projection requires encoder_hidden_size" do
      assert_raise ArgumentError, ~r/encoder_hidden_size required/, fn ->
        Config.new(prefix_projection: true)
      end
    end

    test "validates num_virtual_tokens is positive" do
      assert_raise ArgumentError, ~r/num_virtual_tokens must be at least 1/, fn ->
        Config.new(num_virtual_tokens: 0)
      end
    end

    test "validates encoder_hidden_size is positive when provided" do
      assert_raise ArgumentError, ~r/encoder_hidden_size must be positive/, fn ->
        Config.new(prefix_projection: true, encoder_hidden_size: 0)
      end
    end

    test "sets peft_version" do
      config = Config.new()
      assert config.peft_version == HfPeftEx.version()
    end

    test "accepts base model config fields" do
      config =
        Config.new(
          task_type: :causal_lm,
          base_model_name_or_path: "gpt2",
          inference_mode: true
        )

      assert config.task_type == :causal_lm
      assert config.base_model_name_or_path == "gpt2"
      assert config.inference_mode == true
    end

    test "accepts num_layers and num_attention_heads" do
      config =
        Config.new(
          num_layers: 12,
          num_attention_heads: 12,
          token_dim: 768
        )

      assert config.num_layers == 12
      assert config.num_attention_heads == 12
      assert config.token_dim == 768
    end
  end

  describe "prefix_embedding_dim/1" do
    test "calculates total embedding dimension" do
      config =
        Config.new(
          num_layers: 12,
          token_dim: 768,
          num_virtual_tokens: 20
        )

      # num_layers * 2 (k+v) * token_dim
      assert Config.prefix_embedding_dim(config) == 12 * 2 * 768
    end

    test "returns nil when num_layers is not set" do
      config = Config.new(token_dim: 768)
      assert Config.prefix_embedding_dim(config) == nil
    end

    test "returns nil when token_dim is not set" do
      config = Config.new(num_layers: 12)
      assert Config.prefix_embedding_dim(config) == nil
    end
  end

  describe "trainable_params/1" do
    test "returns trainable params without projection" do
      config =
        Config.new(
          num_virtual_tokens: 20,
          num_layers: 12,
          token_dim: 768,
          prefix_projection: false
        )

      # num_virtual_tokens * num_layers * 2 * token_dim
      expected = 20 * 12 * 2 * 768
      assert Config.trainable_params(config) == expected
    end

    test "returns trainable params with projection" do
      config =
        Config.new(
          num_virtual_tokens: 20,
          num_layers: 12,
          token_dim: 768,
          prefix_projection: true,
          encoder_hidden_size: 512
        )

      # With projection:
      # embedding: num_virtual_tokens * token_dim
      # MLP layer 1: token_dim * encoder_hidden_size + encoder_hidden_size (bias)
      # MLP layer 2: encoder_hidden_size * (num_layers * 2 * token_dim) + (num_layers * 2 * token_dim) (bias)
      embedding_params = 20 * 768
      layer1_params = 768 * 512 + 512
      output_dim = 12 * 2 * 768
      layer2_params = 512 * output_dim + output_dim

      expected = embedding_params + layer1_params + layer2_params
      assert Config.trainable_params(config) == expected
    end

    test "returns nil when required dimensions are missing" do
      config = Config.new()
      assert Config.trainable_params(config) == nil
    end
  end

  describe "prompt_learning?/1" do
    test "returns true" do
      config = Config.new()
      assert Config.prompt_learning?(config) == true
    end
  end

  describe "to_map/1" do
    test "converts config to map" do
      config =
        Config.new(
          num_virtual_tokens: 20,
          prefix_projection: true,
          encoder_hidden_size: 512
        )

      map = Config.to_map(config)

      assert is_map(map)
      assert map.peft_type == "PREFIX_TUNING"
      assert map.num_virtual_tokens == 20
      assert map.prefix_projection == true
      assert map.encoder_hidden_size == 512
    end
  end

  describe "from_map/1" do
    test "creates config from map with atom keys" do
      map = %{
        peft_type: :prefix_tuning,
        num_virtual_tokens: 30,
        prefix_projection: true,
        encoder_hidden_size: 256
      }

      config = Config.from_map(map)

      assert config.peft_type == :prefix_tuning
      assert config.num_virtual_tokens == 30
      assert config.prefix_projection == true
      assert config.encoder_hidden_size == 256
    end

    test "creates config from map with string keys" do
      map = %{
        "peft_type" => "PREFIX_TUNING",
        "num_virtual_tokens" => 30,
        "prefix_projection" => false
      }

      config = Config.from_map(map)

      assert config.peft_type == :prefix_tuning
      assert config.num_virtual_tokens == 30
      assert config.prefix_projection == false
    end
  end

  describe "to_json/1 and from_json/1" do
    test "round-trips through JSON" do
      original =
        Config.new(
          num_virtual_tokens: 25,
          num_layers: 6,
          token_dim: 512,
          num_attention_heads: 8,
          prefix_projection: true,
          encoder_hidden_size: 256
        )

      json = Config.to_json(original)
      restored = Config.from_json(json)

      assert restored.num_virtual_tokens == original.num_virtual_tokens
      assert restored.num_layers == original.num_layers
      assert restored.token_dim == original.token_dim
      assert restored.num_attention_heads == original.num_attention_heads
      assert restored.prefix_projection == original.prefix_projection
      assert restored.encoder_hidden_size == original.encoder_hidden_size
      assert restored.peft_type == :prefix_tuning
    end
  end

  describe "save_pretrained/2 and from_pretrained/1" do
    @tag :tmp_dir
    test "saves and loads config from directory", %{tmp_dir: tmp_dir} do
      config =
        Config.new(
          num_virtual_tokens: 20,
          num_layers: 12,
          token_dim: 768,
          prefix_projection: true,
          encoder_hidden_size: 512
        )

      assert :ok = Config.save_pretrained(config, tmp_dir)

      assert File.exists?(Path.join(tmp_dir, "adapter_config.json"))

      {:ok, loaded} = Config.from_pretrained(tmp_dir)

      assert loaded.num_virtual_tokens == config.num_virtual_tokens
      assert loaded.num_layers == config.num_layers
      assert loaded.token_dim == config.token_dim
      assert loaded.prefix_projection == config.prefix_projection
      assert loaded.encoder_hidden_size == config.encoder_hidden_size
      assert loaded.peft_type == :prefix_tuning
    end
  end
end
