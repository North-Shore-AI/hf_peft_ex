defmodule HfPeftExTest do
  use ExUnit.Case
  doctest HfPeftEx

  alias HfPeftEx.{PeftType, TaskType, Config}
  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig

  describe "HfPeftEx" do
    test "version returns current version" do
      assert HfPeftEx.version() == "0.1.0"
    end
  end

  describe "PeftType" do
    test "all returns all PEFT types" do
      types = PeftType.all()
      assert :lora in types
      assert :adalora in types
      assert :ia3 in types
      assert length(types) == 31
    end

    test "valid? returns true for valid types" do
      assert PeftType.valid?(:lora)
      assert PeftType.valid?(:prefix_tuning)
      refute PeftType.valid?(:invalid)
    end

    test "from_string converts string to atom" do
      assert PeftType.from_string("LORA") == {:ok, :lora}
      assert PeftType.from_string("lora") == {:ok, :lora}
      assert {:error, _} = PeftType.from_string("invalid")
    end

    test "prompt_learning? identifies prompt learning methods" do
      assert PeftType.prompt_learning?(:prefix_tuning)
      assert PeftType.prompt_learning?(:prompt_tuning)
      refute PeftType.prompt_learning?(:lora)
    end
  end

  describe "TaskType" do
    test "all returns all task types" do
      types = TaskType.all()
      assert :causal_lm in types
      assert :seq_cls in types
      assert length(types) == 6
    end

    test "valid? returns true for valid types" do
      assert TaskType.valid?(:causal_lm)
      refute TaskType.valid?(:invalid)
    end
  end

  describe "Config" do
    test "new creates a valid config" do
      config = Config.new(peft_type: :lora, task_type: :causal_lm)
      assert config.peft_type == :lora
      assert config.task_type == :causal_lm
      assert config.peft_version == "0.1.0"
    end

    test "validates invalid peft_type" do
      assert_raise ArgumentError, fn ->
        Config.new(peft_type: :invalid)
      end
    end

    test "to_map converts config to map" do
      config = Config.new(peft_type: :lora)
      map = Config.to_map(config)
      assert is_map(map)
      assert map.peft_type == :lora
    end

    test "prompt_learning? returns correct value" do
      assert Config.prompt_learning?(Config.new(peft_type: :prefix_tuning))
      refute Config.prompt_learning?(Config.new(peft_type: :lora))
    end
  end

  describe "LoraConfig" do
    test "new creates a valid LoRA config with defaults" do
      config = LoraConfig.new()
      assert config.peft_type == :lora
      assert config.r == 8
      assert config.lora_alpha == 8
      assert config.lora_dropout == 0.0
      assert config.bias == :none
    end

    test "new accepts custom values" do
      config =
        LoraConfig.new(
          r: 16,
          lora_alpha: 32,
          target_modules: ["q_proj", "v_proj"],
          task_type: :causal_lm
        )

      assert config.r == 16
      assert config.lora_alpha == 32
      assert config.target_modules == ["q_proj", "v_proj"]
      assert config.task_type == :causal_lm
    end

    test "validates rank must be positive" do
      assert_raise ArgumentError, fn ->
        LoraConfig.new(r: 0)
      end
    end

    test "validates dropout range" do
      assert_raise ArgumentError, fn ->
        LoraConfig.new(lora_dropout: 1.0)
      end
    end

    test "scaling calculates correct factor" do
      config = LoraConfig.new(r: 8, lora_alpha: 16)
      assert LoraConfig.scaling(config) == 2.0
    end

    test "scaling with rslora uses sqrt" do
      config = LoraConfig.new(r: 4, lora_alpha: 8, use_rslora: true)
      assert LoraConfig.scaling(config) == 8 / :math.sqrt(4)
    end

    test "trainable_params calculates parameter count" do
      config = LoraConfig.new(r: 8)
      # For a 1024x1024 layer with rank 8
      params = LoraConfig.trainable_params(config, 1024, 1024)
      assert params == 8 * (1024 + 1024)
    end

    test "trainable_params adds magnitude for dora" do
      config = LoraConfig.new(r: 8, use_dora: true)
      params = LoraConfig.trainable_params(config, 1024, 1024)
      assert params == 8 * (1024 + 1024) + 1024
    end
  end
end
