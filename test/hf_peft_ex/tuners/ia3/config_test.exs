defmodule HfPeftEx.Tuners.IA3.ConfigTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.IA3.Config

  describe "new/1" do
    test "creates config with defaults" do
      config = Config.new()
      assert config.peft_type == :ia3
      assert config.init_ia3_weights == true
      assert config.fan_in_fan_out == false
      assert config.target_modules == nil
      assert config.feedforward_modules == nil
      assert config.exclude_modules == nil
      assert config.modules_to_save == nil
      assert config.inference_mode == false
    end

    test "accepts target_modules as list" do
      config = Config.new(target_modules: ["q_proj", "v_proj"])
      assert config.target_modules == ["q_proj", "v_proj"]
    end

    test "accepts target_modules as string (regex pattern)" do
      config = Config.new(target_modules: ".*_proj")
      assert config.target_modules == ".*_proj"
    end

    test "accepts feedforward_modules" do
      config =
        Config.new(
          target_modules: ["k_proj", "v_proj", "down_proj"],
          feedforward_modules: ["down_proj"]
        )

      assert config.feedforward_modules == ["down_proj"]
    end

    test "validates feedforward_modules is subset of target_modules" do
      assert_raise ArgumentError, ~r/feedforward_modules.*subset/, fn ->
        Config.new(
          target_modules: ["q_proj", "v_proj"],
          feedforward_modules: ["down_proj"]
        )
      end
    end

    test "allows feedforward_modules when subset of target_modules" do
      config =
        Config.new(
          target_modules: ["q_proj", "v_proj", "down_proj"],
          feedforward_modules: ["down_proj"]
        )

      assert config.feedforward_modules == ["down_proj"]
    end

    test "accepts exclude_modules" do
      config = Config.new(target_modules: ["q_proj", "v_proj"], exclude_modules: ["lm_head"])
      assert config.exclude_modules == ["lm_head"]
    end

    test "accepts modules_to_save" do
      config = Config.new(modules_to_save: ["classifier", "score"])
      assert config.modules_to_save == ["classifier", "score"]
    end

    test "accepts init_ia3_weights false" do
      config = Config.new(init_ia3_weights: false)
      assert config.init_ia3_weights == false
    end

    test "accepts fan_in_fan_out true" do
      config = Config.new(fan_in_fan_out: true)
      assert config.fan_in_fan_out == true
    end

    test "accepts task_type" do
      config = Config.new(task_type: :causal_lm)
      assert config.task_type == :causal_lm
    end

    test "sets peft_version" do
      config = Config.new()
      assert config.peft_version != nil
    end

    test "skips feedforward validation when target_modules is nil" do
      config = Config.new(feedforward_modules: ["down_proj"])
      assert config.feedforward_modules == ["down_proj"]
    end

    test "skips feedforward validation when feedforward_modules is nil" do
      config = Config.new(target_modules: ["q_proj", "v_proj"])
      assert config.feedforward_modules == nil
    end

    test "skips feedforward validation when target_modules is regex string" do
      config =
        Config.new(
          target_modules: ".*_proj",
          feedforward_modules: ["down_proj"]
        )

      assert config.feedforward_modules == ["down_proj"]
    end
  end

  describe "to_map/1" do
    test "converts config to map" do
      config = Config.new(target_modules: ["q_proj"], init_ia3_weights: false)
      map = Config.to_map(config)

      assert is_map(map)
      assert map.target_modules == ["q_proj"]
      assert map.init_ia3_weights == false
    end

    test "converts atom values to uppercase strings" do
      config = Config.new(task_type: :causal_lm)
      map = Config.to_map(config)

      assert map.peft_type == "IA3"
      assert map.task_type == "CAUSAL_LM"
    end
  end

  describe "from_map/1" do
    test "creates config from map with atom keys" do
      map = %{target_modules: ["q_proj"], init_ia3_weights: false}
      config = Config.from_map(map)

      assert config.target_modules == ["q_proj"]
      assert config.init_ia3_weights == false
    end

    test "creates config from map with string keys" do
      map = %{"target_modules" => ["q_proj"], "init_ia3_weights" => false}
      config = Config.from_map(map)

      assert config.target_modules == ["q_proj"]
      assert config.init_ia3_weights == false
    end

    test "converts peft_type string to atom" do
      map = %{"peft_type" => "IA3"}
      config = Config.from_map(map)

      assert config.peft_type == :ia3
    end

    test "converts task_type string to atom" do
      map = %{"task_type" => "CAUSAL_LM"}
      config = Config.from_map(map)

      assert config.task_type == :causal_lm
    end
  end

  describe "JSON round trip" do
    test "to_map/1 and from_map/1 round trip preserves data" do
      config =
        Config.new(
          target_modules: ["q_proj", "v_proj"],
          feedforward_modules: ["q_proj"],
          init_ia3_weights: false,
          fan_in_fan_out: true
        )

      map = Config.to_map(config)
      decoded = Config.from_map(map)

      assert decoded.target_modules == config.target_modules
      assert decoded.feedforward_modules == config.feedforward_modules
      assert decoded.init_ia3_weights == config.init_ia3_weights
      assert decoded.fan_in_fan_out == config.fan_in_fan_out
      assert decoded.peft_type == config.peft_type
    end
  end

  describe "save_pretrained/2 and from_pretrained/1" do
    @tag :tmp_dir
    test "saves and loads config", %{tmp_dir: tmp_dir} do
      config =
        Config.new(
          target_modules: ["q_proj", "v_proj"],
          init_ia3_weights: true
        )

      assert :ok == Config.save_pretrained(config, tmp_dir)

      {:ok, loaded} = Config.from_pretrained(tmp_dir)
      assert loaded.target_modules == ["q_proj", "v_proj"]
      assert loaded.init_ia3_weights == true
      assert loaded.peft_type == :ia3
    end
  end

  describe "trainable_params/2" do
    test "returns dimension for non-feedforward layer" do
      config = Config.new()
      params = Config.trainable_params(config, 1024, 768, false)
      # Non-feedforward uses out_features
      assert params == 768
    end

    test "returns dimension for feedforward layer" do
      config = Config.new()
      params = Config.trainable_params(config, 1024, 768, true)
      # Feedforward uses in_features
      assert params == 1024
    end
  end
end
