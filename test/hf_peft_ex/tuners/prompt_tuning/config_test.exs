defmodule HfPeftEx.Tuners.PromptTuning.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.PromptTuning.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :prompt_tuning
      assert config.num_virtual_tokens == 20
      assert config.prompt_tuning_init == :random
      assert config.inference_mode == false
    end

    test "accepts num_virtual_tokens" do
      config = Config.new(num_virtual_tokens: 50)
      assert config.num_virtual_tokens == 50
    end

    test "accepts token_dim" do
      config = Config.new(token_dim: 768)
      assert config.token_dim == 768
    end

    test "accepts prompt_tuning_init as atom" do
      config = Config.new(prompt_tuning_init: :sample_vocab)
      assert config.prompt_tuning_init == :sample_vocab
    end

    test "accepts task_type" do
      config = Config.new(task_type: :causal_lm)
      assert config.task_type == :causal_lm
    end

    test "validates text init requires init_text" do
      assert_raise ArgumentError, ~r/prompt_tuning_init_text required/, fn ->
        Config.new(
          prompt_tuning_init: :text,
          tokenizer_name_or_path: "t5-base"
        )
      end
    end

    test "validates text init requires tokenizer" do
      assert_raise ArgumentError, ~r/tokenizer_name_or_path required/, fn ->
        Config.new(
          prompt_tuning_init: :text,
          prompt_tuning_init_text: "Classify this text:"
        )
      end
    end

    test "allows text init with both init_text and tokenizer" do
      config =
        Config.new(
          prompt_tuning_init: :text,
          prompt_tuning_init_text: "Classify this text:",
          tokenizer_name_or_path: "t5-base"
        )

      assert config.prompt_tuning_init == :text
      assert config.prompt_tuning_init_text == "Classify this text:"
      assert config.tokenizer_name_or_path == "t5-base"
    end

    test "validates num_virtual_tokens > 0" do
      assert_raise ArgumentError, ~r/num_virtual_tokens must be at least 1/, fn ->
        Config.new(num_virtual_tokens: 0)
      end
    end

    test "validates num_virtual_tokens negative" do
      assert_raise ArgumentError, ~r/num_virtual_tokens must be at least 1/, fn ->
        Config.new(num_virtual_tokens: -5)
      end
    end

    test "sets num_transformer_submodules default to 1" do
      config = Config.new()
      assert config.num_transformer_submodules == 1
    end
  end

  describe "prompt_learning?/1" do
    test "returns true" do
      config = Config.new()
      assert Config.prompt_learning?(config) == true
    end
  end

  describe "total_virtual_tokens/1" do
    test "returns num_virtual_tokens * num_transformer_submodules" do
      config = Config.new(num_virtual_tokens: 10, num_transformer_submodules: 2)
      assert Config.total_virtual_tokens(config) == 20
    end

    test "with default submodules is same as num_virtual_tokens" do
      config = Config.new(num_virtual_tokens: 15)
      assert Config.total_virtual_tokens(config) == 15
    end
  end

  describe "trainable_params/1" do
    test "calculates parameter count" do
      config = Config.new(num_virtual_tokens: 20, token_dim: 768)
      # 20 * 768 = 15,360 parameters
      assert Config.trainable_params(config) == 15_360
    end

    test "accounts for transformer submodules" do
      config = Config.new(num_virtual_tokens: 20, token_dim: 768, num_transformer_submodules: 2)
      # (20 * 2) * 768 = 30,720 parameters
      assert Config.trainable_params(config) == 30_720
    end

    test "returns nil when token_dim is not set" do
      config = Config.new(num_virtual_tokens: 20)
      assert Config.trainable_params(config) == nil
    end
  end

  describe "to_map/1" do
    test "converts config to map" do
      config = Config.new(num_virtual_tokens: 30, token_dim: 768)
      map = Config.to_map(config)

      assert is_map(map)
      assert map.num_virtual_tokens == 30
      assert map.token_dim == 768
      assert map.peft_type == "PROMPT_TUNING"
      assert map.prompt_tuning_init == "RANDOM"
    end
  end

  describe "from_map/1" do
    test "creates config from map" do
      map = %{
        "peft_type" => "PROMPT_TUNING",
        "num_virtual_tokens" => 30,
        "token_dim" => 768,
        "prompt_tuning_init" => "RANDOM"
      }

      config = Config.from_map(map)

      assert config.peft_type == :prompt_tuning
      assert config.num_virtual_tokens == 30
      assert config.token_dim == 768
      assert config.prompt_tuning_init == :random
    end

    test "handles atom keys" do
      map = %{
        peft_type: :prompt_tuning,
        num_virtual_tokens: 25,
        prompt_tuning_init: :sample_vocab
      }

      config = Config.from_map(map)

      assert config.num_virtual_tokens == 25
      assert config.prompt_tuning_init == :sample_vocab
    end
  end

  describe "JSON serialization" do
    test "round trip preserves config" do
      config =
        Config.new(
          num_virtual_tokens: 30,
          prompt_tuning_init: :random,
          token_dim: 768
        )

      json = Config.to_json(config)
      decoded = Config.from_json(json)

      assert decoded.num_virtual_tokens == 30
      assert decoded.prompt_tuning_init == :random
      assert decoded.token_dim == 768
      assert decoded.peft_type == :prompt_tuning
    end

    test "handles sample_vocab init" do
      config = Config.new(prompt_tuning_init: :sample_vocab, token_dim: 512)

      json = Config.to_json(config)
      decoded = Config.from_json(json)

      assert decoded.prompt_tuning_init == :sample_vocab
    end

    test "handles text init with all fields" do
      config =
        Config.new(
          prompt_tuning_init: :text,
          prompt_tuning_init_text: "Classify:",
          tokenizer_name_or_path: "t5-base"
        )

      json = Config.to_json(config)
      decoded = Config.from_json(json)

      assert decoded.prompt_tuning_init == :text
      assert decoded.prompt_tuning_init_text == "Classify:"
      assert decoded.tokenizer_name_or_path == "t5-base"
    end
  end

  describe "save_pretrained/2 and from_pretrained/1" do
    @tag :tmp_dir
    test "saves and loads config", %{tmp_dir: dir} do
      config =
        Config.new(
          num_virtual_tokens: 25,
          token_dim: 512,
          prompt_tuning_init: :sample_vocab
        )

      :ok = Config.save_pretrained(config, dir)

      assert File.exists?(Path.join(dir, "adapter_config.json"))

      {:ok, loaded} = Config.from_pretrained(dir)

      assert loaded.num_virtual_tokens == 25
      assert loaded.token_dim == 512
      assert loaded.prompt_tuning_init == :sample_vocab
      assert loaded.peft_type == :prompt_tuning
    end
  end
end
