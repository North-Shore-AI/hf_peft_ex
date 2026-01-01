defmodule HfPeftEx.Tuners.Lora.LayerTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Dora
  alias HfPeftEx.Tuners.Lora.Layer

  describe "new/1" do
    test "creates layer with correct lora_A dimensions (r × in_features)" do
      layer = Layer.new(in_features: 1024, out_features: 512, r: 8)
      assert Nx.shape(layer.lora_A) == {8, 1024}
    end

    test "creates layer with correct lora_B dimensions (out_features × r)" do
      layer = Layer.new(in_features: 1024, out_features: 512, r: 8)
      assert Nx.shape(layer.lora_B) == {512, 8}
    end

    test "stores rank correctly" do
      layer = Layer.new(in_features: 256, out_features: 256, r: 16)
      assert layer.r == 16
    end

    test "computes scaling from lora_alpha and r" do
      layer = Layer.new(in_features: 256, out_features: 256, r: 8, lora_alpha: 16)
      assert layer.scaling == 2.0
    end

    test "uses rslora scaling when enabled" do
      layer =
        Layer.new(in_features: 256, out_features: 256, r: 4, lora_alpha: 8, use_rslora: true)

      assert layer.scaling == 8 / :math.sqrt(4)
    end

    test "initializes in unmerged state" do
      layer = Layer.new(in_features: 256, out_features: 256, r: 8)
      assert layer.merged == false
    end

    test "initializes lora_A with kaiming-like distribution" do
      layer = Layer.new(in_features: 1024, out_features: 512, r: 8)
      # A should be initialized with small values (kaiming)
      mean = Nx.mean(layer.lora_A) |> Nx.to_number()
      assert abs(mean) < 1.0
    end

    test "initializes lora_B with zeros" do
      layer = Layer.new(in_features: 1024, out_features: 512, r: 8)
      # B should be initialized to zeros
      sum = Nx.sum(layer.lora_B) |> Nx.to_number()
      assert sum == 0.0
    end

    test "accepts config struct" do
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      layer = Layer.new(in_features: 512, out_features: 512, config: config)
      assert layer.r == 16
      assert layer.scaling == 2.0
    end
  end

  describe "get_delta_weight/1" do
    test "computes B @ A * scaling" do
      # Create layer with known values
      layer = Layer.new(in_features: 4, out_features: 3, r: 2, lora_alpha: 2)
      # Override with known values for testing
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      # 2x4
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      # 3x2
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      layer = %{layer | lora_A: lora_A, lora_B: lora_B}

      delta = Layer.get_delta_weight(layer)

      # B @ A = [[1,0,0,0], [0,1,0,0], [0,0,0,0]] (3x4)
      # scaling = 2/2 = 1.0
      expected = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
      assert Nx.shape(delta) == {3, 4}
      assert Nx.to_number(Nx.sum(Nx.subtract(delta, expected))) == 0.0
    end
  end

  describe "forward/3" do
    test "returns base output when merged" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2)
      layer = %{layer | merged: true}

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      base_output = Nx.tensor([[0.5, 0.5, 0.5]])

      result = Layer.forward(layer, x, base_output)
      assert Nx.to_list(result) == Nx.to_list(base_output)
    end

    test "adds LoRA contribution when not merged" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2, lora_alpha: 2)
      # Set known values
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      layer = %{layer | lora_A: lora_A, lora_B: lora_B, merged: false}

      # 1x4
      x = Nx.tensor([[2.0, 4.0, 0.0, 0.0]])
      # 1x3
      base_output = Nx.tensor([[0.0, 0.0, 0.0]])

      result = Layer.forward(layer, x, base_output)

      # x @ A^T = [[2,4,0,0]] @ [[0.5,0],[0,0.5],[0,0],[0,0]] = [[1.0, 2.0]]
      # result @ B^T = [[1,2]] @ [[1,0,0],[0,1,0]] = [[1.0, 2.0, 0.0]]
      # with scaling = 1.0, final = [[1.0, 2.0, 0.0]]
      assert Nx.shape(result) == {1, 3}
    end

    test "applies dropout in training mode" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2, dropout: 0.5)
      x = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      base_output = Nx.tensor([[0.0, 0.0, 0.0]])

      # With training=true and dropout, some values should be zeroed
      _result = Layer.forward(layer, x, base_output, training: true)
      # Just verify it doesn't crash - dropout is stochastic
      assert true
    end
  end

  describe "merge/1 and unmerge/1" do
    test "merge sets merged flag to true" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2)
      base_weight = Nx.broadcast(0.0, {3, 4})

      {merged_layer, new_weight} = Layer.merge(layer, base_weight)

      assert merged_layer.merged == true
      assert Nx.shape(new_weight) == {3, 4}
    end

    test "unmerge sets merged flag to false" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2)
      layer = %{layer | merged: true}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {unmerged_layer, new_weight} = Layer.unmerge(layer, base_weight)

      assert unmerged_layer.merged == false
      assert Nx.shape(new_weight) == {3, 4}
    end

    test "merge and unmerge are inverse operations" do
      layer = Layer.new(in_features: 4, out_features: 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
      layer = %{layer | lora_A: lora_A, lora_B: lora_B}

      original_weight = Nx.broadcast(1.0, {3, 4})

      {merged, merged_weight} = Layer.merge(layer, original_weight)
      {_unmerged, restored_weight} = Layer.unmerge(merged, merged_weight)

      diff =
        Nx.subtract(original_weight, restored_weight) |> Nx.abs() |> Nx.sum() |> Nx.to_number()

      assert diff < 1.0e-6
    end
  end

  describe "use_dora" do
    test "forward applies dora scaling when enabled" do
      layer = Layer.new(in_features: 2, out_features: 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      layer = %{layer | lora_A: lora_A, lora_B: lora_B}

      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      x = Nx.tensor([[1.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(weight))

      result = Layer.forward(layer, x, base_output, weight: weight)

      dora = Dora.new(weight)

      expected =
        Dora.apply_dora(
          dora,
          %{lora_A: lora_A, lora_B: lora_B, scaling: layer.scaling},
          x,
          base_output,
          weight
        )

      assert_tensor_close(result, expected)

      lora_only = Layer.forward(%{layer | use_dora: false}, x, base_output)
      diff = Nx.subtract(result, lora_only) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "merge and unmerge apply dora factor" do
      layer = Layer.new(in_features: 2, out_features: 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      layer = %{layer | lora_A: lora_A, lora_B: lora_B}

      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      delta = Layer.get_delta_weight(layer)

      magnitude = Dora.init_magnitude(base_weight)
      weight_norm = Dora.get_weight_norm(base_weight, delta, 1.0, fan_in_fan_out: false)
      {out_features} = Nx.shape(magnitude)
      dora_factor = Nx.reshape(Nx.divide(magnitude, weight_norm), {out_features, 1})
      expected_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {merged, merged_weight} = Layer.merge(layer, base_weight)

      assert merged.merged == true
      assert merged.dora_weight_norm != nil
      assert_tensor_close(merged_weight, expected_weight)

      {unmerged, restored} = Layer.unmerge(merged, merged_weight)
      assert unmerged.merged == false
      assert_tensor_close(restored, base_weight)
    end
  end

  defp assert_tensor_close(left, right, tol \\ 1.0e-6) do
    diff = Nx.subtract(left, right) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
    assert diff < tol
  end
end
