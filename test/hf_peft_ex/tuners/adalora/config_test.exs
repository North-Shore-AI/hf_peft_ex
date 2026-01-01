defmodule HfPeftEx.Tuners.Adalora.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new(total_step: 1000)
      assert config.peft_type == :adalora
      assert config.init_r == 12
      assert config.target_r == 8
      assert config.tinit == 0
      assert config.tfinal == 0
      assert config.delta_t == 1
      assert config.beta1 == 0.85
      assert config.beta2 == 0.85
      assert config.orth_reg_weight == 0.5
    end

    test "validates total_step is required" do
      assert_raise ArgumentError, ~r/total_step is required/, fn ->
        Config.new()
      end
    end

    test "validates total_step > 0" do
      assert_raise ArgumentError, ~r/total_step.*must be positive/, fn ->
        Config.new(total_step: 0)
      end
    end

    test "validates total_step cannot be negative" do
      assert_raise ArgumentError, ~r/total_step.*must be positive/, fn ->
        Config.new(total_step: -10)
      end
    end

    test "validates tinit < total_step - tfinal" do
      assert_raise ArgumentError, ~r/tinit must be less than/, fn ->
        Config.new(total_step: 100, tinit: 80, tfinal: 30)
      end
    end

    test "validates tinit equals boundary" do
      assert_raise ArgumentError, ~r/tinit must be less than/, fn ->
        Config.new(total_step: 100, tinit: 50, tfinal: 50)
      end
    end

    test "validates init_r >= target_r" do
      assert_raise ArgumentError, ~r/init_r must be >= target_r/, fn ->
        Config.new(total_step: 100, init_r: 4, target_r: 8)
      end
    end

    test "allows custom config values" do
      config =
        Config.new(
          total_step: 5000,
          init_r: 16,
          target_r: 4,
          tinit: 100,
          tfinal: 200,
          delta_t: 10,
          beta1: 0.9,
          beta2: 0.95,
          orth_reg_weight: 0.3
        )

      assert config.total_step == 5000
      assert config.init_r == 16
      assert config.target_r == 4
      assert config.tinit == 100
      assert config.tfinal == 200
      assert config.delta_t == 10
      assert config.beta1 == 0.9
      assert config.beta2 == 0.95
      assert config.orth_reg_weight == 0.3
    end
  end

  describe "inherits LoRA config fields" do
    test "has lora_alpha and lora_dropout" do
      config = Config.new(total_step: 100, lora_alpha: 32, lora_dropout: 0.1)
      assert config.lora_alpha == 32
      assert config.lora_dropout == 0.1
    end

    test "has target_modules" do
      config = Config.new(total_step: 100, target_modules: ["q_proj", "v_proj"])
      assert config.target_modules == ["q_proj", "v_proj"]
    end

    test "has bias configuration" do
      config = Config.new(total_step: 100, bias: :lora_only)
      assert config.bias == :lora_only
    end

    test "has task_type" do
      config = Config.new(total_step: 100, task_type: :causal_lm)
      assert config.task_type == :causal_lm
    end
  end

  describe "validates DoRA is not supported" do
    test "raises when use_dora is true" do
      assert_raise ArgumentError, ~r/AdaLoRA does not support DoRA/, fn ->
        Config.new(total_step: 100, use_dora: true)
      end
    end
  end

  describe "scaling/1" do
    test "returns lora_alpha / r style scaling" do
      config = Config.new(total_step: 100, lora_alpha: 16, init_r: 8)
      # AdaLoRA scaling uses init_r for the initial rank
      assert Config.scaling(config) == 2.0
    end
  end
end
