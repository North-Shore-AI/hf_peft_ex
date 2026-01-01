defmodule HfPeftEx.MappingTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Mapping

  describe "get_config_class/1" do
    test "returns LoRA config for :lora" do
      assert Mapping.get_config_class(:lora) == HfPeftEx.Tuners.Lora.Config
    end

    test "returns nil for unknown type" do
      assert Mapping.get_config_class(:unknown) == nil
    end

    test "returns config class for all known types" do
      # Only test types that have implementations
      known_types = [:lora, :adalora, :ia3, :prefix_tuning, :prompt_tuning]

      for type <- known_types do
        assert Mapping.get_config_class(type) != nil,
               "Expected config class for #{inspect(type)} but got nil"
      end
    end
  end

  describe "get_tuner_class/1" do
    test "returns LoRA model for :lora" do
      assert Mapping.get_tuner_class(:lora) == HfPeftEx.Tuners.Lora.Model
    end

    test "returns nil for unknown type" do
      assert Mapping.get_tuner_class(:unknown) == nil
    end
  end

  describe "get_prefix/1" do
    test "returns lora_ for :lora" do
      assert Mapping.get_prefix(:lora) == "lora_"
    end

    test "returns ia3_ for :ia3" do
      assert Mapping.get_prefix(:ia3) == "ia3_"
    end

    test "returns nil for unknown type" do
      assert Mapping.get_prefix(:unknown) == nil
    end

    test "all prefixes end with underscore" do
      for type <- Mapping.supported_peft_types() do
        prefix = Mapping.get_prefix(type)
        assert String.ends_with?(prefix, "_"), "Prefix for #{type} should end with underscore"
      end
    end
  end

  describe "get_peft_config/1" do
    test "creates config from dict with peft_type as string" do
      dict = %{
        "peft_type" => "lora",
        "r" => 8,
        "lora_alpha" => 16
      }

      {:ok, config} = Mapping.get_peft_config(dict)

      assert config.__struct__ == HfPeftEx.Tuners.Lora.Config
      assert config.r == 8
      assert config.lora_alpha == 16
    end

    test "creates config from dict with peft_type as atom" do
      dict = %{
        "peft_type" => :lora,
        "r" => 16,
        "lora_alpha" => 32
      }

      {:ok, config} = Mapping.get_peft_config(dict)

      assert config.__struct__ == HfPeftEx.Tuners.Lora.Config
      assert config.r == 16
    end

    test "returns error for missing peft_type" do
      assert {:error, reason} = Mapping.get_peft_config(%{"r" => 8})
      assert reason =~ "peft_type"
    end

    test "returns error for unknown peft_type" do
      assert {:error, reason} = Mapping.get_peft_config(%{"peft_type" => "unknown"})
      assert reason =~ "unknown" or reason =~ "Unknown"
    end

    test "handles atom keys in dict" do
      dict = %{
        peft_type: "lora",
        r: 8,
        lora_alpha: 16
      }

      {:ok, config} = Mapping.get_peft_config(dict)
      assert config.r == 8
    end
  end

  describe "supported_peft_types/0" do
    test "returns list of all supported types" do
      types = Mapping.supported_peft_types()

      assert :lora in types
      assert is_list(types)
      assert types != []
    end

    test "all returned types are valid PeftType atoms" do
      types = Mapping.supported_peft_types()

      for type <- types do
        assert HfPeftEx.PeftType.valid?(type),
               "#{inspect(type)} should be a valid PeftType"
      end
    end
  end

  describe "inject_adapter_in_model/3" do
    test "returns error for prompt learning methods" do
      config = %HfPeftEx.Config{peft_type: :prefix_tuning}

      assert {:error, reason} = Mapping.inject_adapter_in_model(config, %{}, "default")
      assert reason =~ "prompt learning" or reason =~ "not supported"
    end

    test "returns error for unknown peft type" do
      config = %HfPeftEx.Config{peft_type: :unknown_type}

      assert {:error, _reason} = Mapping.inject_adapter_in_model(config, %{}, "default")
    end
  end
end
