defmodule HfPeftEx.Utils.SaveAndLoadTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Utils.SaveAndLoad

  # Helper to generate a random tensor with the correct Nx.Random API
  defp random_tensor(shape, opts \\ []) do
    type = Keyword.get(opts, :type, :f32)
    key = Nx.Random.key(System.unique_integer([:positive]))
    {tensor, _new_key} = Nx.Random.normal(key, shape: shape, type: type)
    tensor
  end

  # Helper to create a mock LoRA model for testing
  defp create_mock_lora_model(opts \\ []) do
    bias_mode = Keyword.get(opts, :bias, :none)
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    # Create mock LoRA A and B weights
    lora_a = random_tensor({4, 32})
    lora_b = random_tensor({64, 4})

    lora_layers = %{
      "layer1" => %{
        lora_a: %{adapter_name => lora_a},
        lora_b: %{adapter_name => lora_b},
        lora_bias:
          if bias_mode != :none do
            %{adapter_name => Nx.broadcast(Nx.tensor(0.0), {64})}
          else
            nil
          end
      }
    }

    config = %HfPeftEx.Tuners.Lora.Config{
      peft_type: :lora,
      bias: bias_mode,
      r: 4,
      lora_alpha: 8
    }

    %{
      lora_layers: lora_layers,
      config: config,
      active_adapter: adapter_name,
      peft_config: %{adapter_name => config}
    }
  end

  describe "get_peft_model_state_dict/3" do
    test "extracts adapter weights only" do
      model = create_mock_lora_model()

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # Should only have lora keys
      for key <- Map.keys(state_dict) do
        assert String.contains?(key, "lora_"),
               "Key #{key} should contain 'lora_'"
      end
    end

    test "includes lora_A and lora_B weights" do
      model = create_mock_lora_model()

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      lora_a_keys = Enum.filter(Map.keys(state_dict), &String.contains?(&1, "lora_A"))
      lora_b_keys = Enum.filter(Map.keys(state_dict), &String.contains?(&1, "lora_B"))

      assert lora_a_keys != [], "Should have lora_A keys"
      assert lora_b_keys != [], "Should have lora_B keys"
    end

    test "removes adapter name prefix from keys" do
      model = create_mock_lora_model()

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model, "default")

      # Keys should not include adapter name in the middle
      for key <- Map.keys(state_dict) do
        refute String.contains?(key, ".default."),
               "Key #{key} should not contain adapter name"
      end
    end

    test "handles bias mode :none" do
      model = create_mock_lora_model(bias: :none)

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # Should not include any bias keys
      bias_keys = Enum.filter(Map.keys(state_dict), &String.contains?(&1, "bias"))
      assert Enum.empty?(bias_keys), "Should not have bias keys in :none mode"
    end

    test "handles bias mode :all" do
      model = create_mock_lora_model(bias: :all)

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # State dict keys should exist
      assert map_size(state_dict) > 0
    end

    test "handles bias mode :lora_only" do
      model = create_mock_lora_model(bias: :lora_only)

      {:ok, state_dict} = SaveAndLoad.get_peft_model_state_dict(model)

      # Should include lora bias if present
      assert map_size(state_dict) > 0
    end
  end

  describe "set_peft_model_state_dict/3" do
    test "loads weights into model" do
      model = create_mock_lora_model()

      state_dict = %{
        "layer1.lora_A.weight" => random_tensor({4, 32}),
        "layer1.lora_B.weight" => random_tensor({64, 4})
      }

      {:ok, updated} = SaveAndLoad.set_peft_model_state_dict(model, state_dict)

      # Verify weights were loaded
      assert updated.lora_layers["layer1"].lora_a["default"] != nil
    end

    test "adds adapter name prefix to keys" do
      model = create_mock_lora_model()

      state_dict = %{
        "layer1.lora_A.weight" => random_tensor({4, 32})
      }

      {:ok, updated} = SaveAndLoad.set_peft_model_state_dict(model, state_dict, "custom")

      assert updated.lora_layers["layer1"].lora_a["custom"] != nil
    end

    test "handles missing layers gracefully" do
      model = create_mock_lora_model()

      state_dict = %{
        "nonexistent_layer.lora_A.weight" => random_tensor({4, 32})
      }

      {:ok, updated} = SaveAndLoad.set_peft_model_state_dict(model, state_dict)

      # Original layers should be unchanged
      assert updated.lora_layers["layer1"] != nil
    end

    test "handles mismatched sizes with ignore option" do
      model = create_mock_lora_model()

      # Wrong shape (8 instead of 4 for rank)
      state_dict = %{
        "layer1.lora_A.weight" => random_tensor({8, 32})
      }

      result =
        SaveAndLoad.set_peft_model_state_dict(
          model,
          state_dict,
          "default",
          ignore_mismatched_sizes: true
        )

      assert {:ok, _} = result
    end
  end

  describe "save_peft_weights/3" do
    @tag :tmp_dir
    test "saves weights to Nx format file", %{tmp_dir: dir} do
      state_dict = %{
        "layer1.lora_A.weight" => random_tensor({4, 32}),
        "layer1.lora_B.weight" => random_tensor({64, 4})
      }

      path = Path.join(dir, "adapter_model.nx")
      :ok = SaveAndLoad.save_peft_weights(state_dict, path, format: :nx)

      assert File.exists?(path)
    end

    @tag :tmp_dir
    test "saves to default Nx format", %{tmp_dir: dir} do
      state_dict = %{
        "layer1.lora_A.weight" => random_tensor({4, 32})
      }

      path = Path.join(dir, "adapter_model.nx")
      :ok = SaveAndLoad.save_peft_weights(state_dict, path)

      assert File.exists?(path)
    end
  end

  describe "load_peft_weights/2" do
    @tag :tmp_dir
    test "loads weights from file", %{tmp_dir: dir} do
      original = %{
        "layer1.lora_A.weight" => random_tensor({4, 32})
      }

      path = Path.join(dir, "adapter_model.nx")
      :ok = SaveAndLoad.save_peft_weights(original, path, format: :nx)

      {:ok, loaded} = SaveAndLoad.load_peft_weights(path)

      assert Map.has_key?(loaded, "layer1.lora_A.weight")
      assert Nx.shape(loaded["layer1.lora_A.weight"]) == {4, 32}
    end

    @tag :tmp_dir
    test "round-trip preserves tensor values", %{tmp_dir: dir} do
      original_tensor = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      original = %{"test.weight" => original_tensor}

      path = Path.join(dir, "test.nx")
      :ok = SaveAndLoad.save_peft_weights(original, path)

      {:ok, loaded} = SaveAndLoad.load_peft_weights(path)

      assert Nx.equal(loaded["test.weight"], original_tensor) |> Nx.all() |> Nx.to_number() == 1
    end

    test "returns error for missing file" do
      assert {:error, reason} = SaveAndLoad.load_peft_weights("/nonexistent/path")
      assert reason =~ "not found" or reason =~ "enoent" or is_atom(reason)
    end
  end

  describe "filter_adapter_keys/3" do
    test "filters to only adapter-related keys" do
      state_dict = %{
        "base_model.layer1.weight" => Nx.tensor([1.0]),
        "base_model.layer1.lora_A.default.weight" => Nx.tensor([2.0]),
        "base_model.layer1.lora_B.default.weight" => Nx.tensor([3.0])
      }

      filtered = SaveAndLoad.filter_adapter_keys(state_dict, "default", :lora)

      assert map_size(filtered) == 2

      for key <- Map.keys(filtered) do
        assert String.contains?(key, "lora_")
      end
    end

    test "filters by adapter name" do
      state_dict = %{
        "layer1.lora_A.default.weight" => Nx.tensor([1.0]),
        "layer1.lora_A.other.weight" => Nx.tensor([2.0])
      }

      filtered = SaveAndLoad.filter_adapter_keys(state_dict, "default", :lora)

      assert map_size(filtered) == 1
      assert Map.has_key?(filtered, "layer1.lora_A.default.weight")
    end

    test "filters by peft type prefix" do
      state_dict = %{
        "layer1.lora_A.default.weight" => Nx.tensor([1.0]),
        "layer1.ia3_l.default.weight" => Nx.tensor([2.0])
      }

      filtered = SaveAndLoad.filter_adapter_keys(state_dict, "default", :lora)

      assert map_size(filtered) == 1
      assert Map.has_key?(filtered, "layer1.lora_A.default.weight")
    end
  end

  describe "remove_adapter_name_from_keys/2" do
    test "removes adapter name from middle of key" do
      state_dict = %{
        "layer1.lora_A.default.weight" => Nx.tensor([1.0]),
        "layer1.lora_B.default.weight" => Nx.tensor([2.0])
      }

      result = SaveAndLoad.remove_adapter_name_from_keys(state_dict, "default")

      assert Map.has_key?(result, "layer1.lora_A.weight")
      assert Map.has_key?(result, "layer1.lora_B.weight")
    end

    test "handles keys without adapter name" do
      state_dict = %{
        "layer1.weight" => Nx.tensor([1.0])
      }

      result = SaveAndLoad.remove_adapter_name_from_keys(state_dict, "default")

      assert Map.has_key?(result, "layer1.weight")
    end
  end

  describe "add_adapter_name_to_keys/2" do
    test "adds adapter name to keys" do
      state_dict = %{
        "layer1.lora_A.weight" => Nx.tensor([1.0]),
        "layer1.lora_B.weight" => Nx.tensor([2.0])
      }

      result = SaveAndLoad.add_adapter_name_to_keys(state_dict, "custom", "lora_")

      assert Map.has_key?(result, "layer1.lora_A.custom.weight")
      assert Map.has_key?(result, "layer1.lora_B.custom.weight")
    end
  end
end
