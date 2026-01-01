defmodule HfPeftEx.PeftModelTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.PeftModel
  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig

  describe "new/2" do
    test "creates a PEFT model from base model and LoRA config" do
      config = LoraConfig.new(r: 8, lora_alpha: 16)
      model = PeftModel.new(%{type: :transformer}, config)

      assert model.peft_type == :lora
      assert model.adapter_name == "default"
    end

    test "stores base model reference" do
      base_model = %{type: :gpt2, layers: 12}
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(base_model, config)

      assert model.base_model == base_model
    end

    test "accepts custom adapter name" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config, adapter_name: "custom")

      assert model.adapter_name == "custom"
    end
  end

  describe "get_peft_config/1" do
    test "returns the PEFT configuration" do
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      model = PeftModel.new(%{}, config)

      retrieved = PeftModel.get_peft_config(model)
      assert retrieved.r == 16
      assert retrieved.lora_alpha == 32
      assert retrieved.peft_type == :lora
    end

    test "returns config for specific adapter" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config, adapter_name: "my_adapter")

      retrieved = PeftModel.get_peft_config(model, "my_adapter")
      assert retrieved.r == 8
    end
  end

  describe "get_nb_trainable_parameters/1" do
    test "returns trainable and total parameter counts" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config)
      # Mock some parameter counts
      model = %{model | trainable_params: 1000, total_params: 100_000}

      {trainable, total} = PeftModel.get_nb_trainable_parameters(model)
      assert trainable == 1000
      assert total == 100_000
    end
  end

  describe "print_trainable_parameters/1" do
    test "returns formatted string with parameter stats" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config)
      model = %{model | trainable_params: 1024, total_params: 1_000_000}

      output = PeftModel.print_trainable_parameters(model)

      assert output =~ "trainable params"
      assert output =~ "1,024"
      assert output =~ "1,000,000"
      assert output =~ "%"
    end
  end

  describe "save_pretrained/2" do
    @tag :tmp_dir
    test "saves adapter config to directory", %{tmp_dir: tmp_dir} do
      config = LoraConfig.new(r: 8, lora_alpha: 16)
      model = PeftModel.new(%{}, config)

      :ok = PeftModel.save_pretrained(model, tmp_dir)

      config_path = Path.join(tmp_dir, "adapter_config.json")
      assert File.exists?(config_path)

      {:ok, content} = File.read(config_path)
      {:ok, json} = Jason.decode(content)
      assert json["r"] == 8
      assert json["lora_alpha"] == 16
    end
  end

  describe "from_pretrained/2" do
    @tag :tmp_dir
    test "loads adapter from directory", %{tmp_dir: tmp_dir} do
      # First save
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      model = PeftModel.new(%{type: :test}, config)
      :ok = PeftModel.save_pretrained(model, tmp_dir)

      # Then load
      base_model = %{type: :test}
      {:ok, loaded_model} = PeftModel.from_pretrained(base_model, tmp_dir)

      assert loaded_model.base_model == base_model
      loaded_config = PeftModel.get_peft_config(loaded_model)
      assert loaded_config.r == 16
      assert loaded_config.lora_alpha == 32
    end
  end

  describe "active_adapters/1" do
    test "returns list of active adapter names" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config, adapter_name: "my_adapter")

      adapters = PeftModel.active_adapters(model)
      assert adapters == ["my_adapter"]
    end
  end

  describe "set_adapter/2" do
    test "sets the active adapter" do
      config = LoraConfig.new(r: 8)
      model = PeftModel.new(%{}, config)
      model = %{model | peft_configs: %{"adapter1" => config, "adapter2" => config}}

      model = PeftModel.set_adapter(model, "adapter2")
      assert model.active_adapter == "adapter2"
    end
  end
end
