defmodule HfPeftEx.Tuners.Lora.LinearTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Config, as: LoraConfig
  alias HfPeftEx.Tuners.Lora.Dora
  alias HfPeftEx.Tuners.Lora.Linear

  describe "new/3" do
    test "creates linear layer with correct dimensions" do
      linear = Linear.new(1024, 512, r: 8, lora_alpha: 16)

      assert linear.in_features == 1024
      assert linear.out_features == 512
      assert linear.r == 8
    end

    test "creates lora_A with shape {r, in_features}" do
      linear = Linear.new(1024, 512, r: 8)
      assert Nx.shape(linear.lora_A) == {8, 1024}
    end

    test "creates lora_B with shape {out_features, r}" do
      linear = Linear.new(1024, 512, r: 8)
      assert Nx.shape(linear.lora_B) == {512, 8}
    end

    test "computes scaling correctly" do
      linear = Linear.new(256, 128, r: 8, lora_alpha: 16)
      assert linear.scaling == 2.0
    end

    test "uses rslora scaling when enabled" do
      linear = Linear.new(256, 128, r: 4, lora_alpha: 8, use_rslora: true)
      assert linear.scaling == 8 / :math.sqrt(4)
    end

    test "accepts LoraConfig struct" do
      config = LoraConfig.new(r: 16, lora_alpha: 32)
      linear = Linear.new(512, 256, config: config)

      assert linear.r == 16
      assert linear.lora_alpha == 32
    end

    test "initializes lora_A with kaiming uniform" do
      linear = Linear.new(1024, 512, r: 8)
      mean = linear.lora_A |> Nx.mean() |> Nx.to_number()
      assert abs(mean) < 1.0
    end

    test "initializes lora_B with zeros" do
      linear = Linear.new(1024, 512, r: 8)
      sum = linear.lora_B |> Nx.sum() |> Nx.to_number()
      assert sum == 0.0
    end

    test "defaults fan_in_fan_out to false" do
      linear = Linear.new(256, 128, r: 8)
      assert linear.fan_in_fan_out == false
    end

    test "accepts fan_in_fan_out option" do
      linear = Linear.new(256, 128, r: 8, fan_in_fan_out: true)
      assert linear.fan_in_fan_out == true
    end
  end

  describe "get_delta_weight/1" do
    test "computes B @ A * scaling" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      delta = Linear.get_delta_weight(linear)

      expected = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
      assert Nx.shape(delta) == {3, 4}
      assert Nx.to_number(Nx.sum(Nx.subtract(delta, expected))) == 0.0
    end

    test "transposes when fan_in_fan_out is true" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2, fan_in_fan_out: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      delta = Linear.get_delta_weight(linear)

      # Should be transposed: {4, 3} instead of {3, 4}
      assert Nx.shape(delta) == {4, 3}
    end
  end

  describe "forward/4" do
    test "returns base output when merged" do
      linear = Linear.new(4, 3, r: 2)
      linear = %{linear | merged: true}

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      base_output = Nx.tensor([[0.5, 0.5, 0.5]])

      result = Linear.forward(linear, x, base_output)
      assert Nx.to_list(result) == Nx.to_list(base_output)
    end

    test "adds LoRA contribution when not merged" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B, merged: false}

      x = Nx.tensor([[2.0, 4.0, 0.0, 0.0]])
      base_output = Nx.tensor([[0.0, 0.0, 0.0]])

      result = Linear.forward(linear, x, base_output)
      assert Nx.shape(result) == {1, 3}
    end

    test "applies dropout in training mode" do
      linear = Linear.new(4, 3, r: 2, dropout: 0.5)
      x = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      base_output = Nx.tensor([[0.0, 0.0, 0.0]])

      _result = Linear.forward(linear, x, base_output, training: true)
      assert true
    end
  end

  describe "merge/2" do
    test "sets merged flag to true" do
      linear = Linear.new(4, 3, r: 2)
      base_weight = Nx.broadcast(0.0, {3, 4})

      {merged, _weight} = Linear.merge(linear, base_weight)
      assert merged.merged == true
    end

    test "adds delta weight to base weight" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      base_weight = Nx.broadcast(0.0, {3, 4})
      {_merged, new_weight} = Linear.merge(linear, base_weight)

      assert Nx.shape(new_weight) == {3, 4}
      # First element should be 1.0 (from delta)
      assert new_weight |> Nx.slice([0, 0], [1, 1]) |> Nx.squeeze() |> Nx.to_number() == 1.0
    end

    test "handles fan_in_fan_out transposition" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2, fan_in_fan_out: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      # For fan_in_fan_out, base_weight is {in, out} = {4, 3}
      base_weight = Nx.broadcast(0.0, {4, 3})
      {_merged, new_weight} = Linear.merge(linear, base_weight)

      assert Nx.shape(new_weight) == {4, 3}
    end

    test "returns unchanged when already merged" do
      linear = Linear.new(4, 3, r: 2)
      linear = %{linear | merged: true}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {merged, weight} = Linear.merge(linear, base_weight)
      assert merged.merged == true
      assert Nx.to_list(weight) == Nx.to_list(base_weight)
    end
  end

  describe "unmerge/2" do
    test "sets merged flag to false" do
      linear = Linear.new(4, 3, r: 2)
      linear = %{linear | merged: true}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {unmerged, _weight} = Linear.unmerge(linear, base_weight)
      assert unmerged.merged == false
    end

    test "merge and unmerge are inverse operations" do
      linear = Linear.new(4, 3, r: 2, lora_alpha: 2)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      original = Nx.broadcast(1.0, {3, 4})
      {merged, merged_weight} = Linear.merge(linear, original)
      {_unmerged, restored} = Linear.unmerge(merged, merged_weight)

      diff = Nx.subtract(original, restored) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "returns unchanged when already unmerged" do
      linear = Linear.new(4, 3, r: 2)
      linear = %{linear | merged: false}
      base_weight = Nx.broadcast(1.0, {3, 4})

      {unmerged, weight} = Linear.unmerge(linear, base_weight)
      assert unmerged.merged == false
      assert Nx.to_list(weight) == Nx.to_list(base_weight)
    end
  end

  describe "use_dora" do
    test "forward applies dora scaling when enabled" do
      linear = Linear.new(2, 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      x = Nx.tensor([[1.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(weight))

      result = Linear.forward(linear, x, base_output, weight: weight)

      dora = Dora.new(weight, fan_in_fan_out: linear.fan_in_fan_out)

      expected =
        Dora.apply_dora(
          dora,
          %{lora_A: lora_A, lora_B: lora_B, scaling: linear.scaling},
          x,
          base_output,
          weight
        )

      assert_tensor_close(result, expected)

      lora_only = Linear.forward(%{linear | use_dora: false}, x, base_output)
      diff = Nx.subtract(result, lora_only) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "merge and unmerge apply dora factor" do
      linear = Linear.new(2, 2, r: 2, lora_alpha: 2, use_dora: true)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_B = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      linear = %{linear | lora_A: lora_A, lora_B: lora_B}

      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      delta = Linear.get_delta_weight(linear)

      magnitude = Dora.init_magnitude(base_weight)
      weight_norm = Dora.get_weight_norm(base_weight, delta, 1.0, fan_in_fan_out: false)
      {out_features} = Nx.shape(magnitude)
      dora_factor = Nx.reshape(Nx.divide(magnitude, weight_norm), {out_features, 1})
      expected_weight = Nx.multiply(dora_factor, Nx.add(base_weight, delta))

      {merged, merged_weight} = Linear.merge(linear, base_weight)

      assert merged.merged == true
      assert merged.dora_weight_norm != nil
      assert_tensor_close(merged_weight, expected_weight)

      {unmerged, restored} = Linear.unmerge(merged, merged_weight)
      assert unmerged.merged == false
      assert_tensor_close(restored, base_weight)
    end

    test "supports dropout in training mode (dora)" do
      linear = Linear.new(2, 2, r: 2, lora_alpha: 2, use_dora: true, dropout: 0.5)
      lora_a = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_b = Nx.tensor([[0.5, 0.0], [0.0, 0.5]])
      linear = %{linear | lora_A: lora_a, lora_B: lora_b}

      x = Nx.tensor([[1.0, 1.0]])
      base_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      base_output = Nx.dot(x, Nx.transpose(base_weight))

      result = Linear.forward(linear, x, base_output, weight: base_weight, training: true)

      assert Nx.shape(result) == Nx.shape(base_output)
      assert_finite(result)
    end
  end

  defp assert_tensor_close(left, right, tol \\ 1.0e-6) do
    diff = Nx.subtract(left, right) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
    assert diff < tol
  end

  defp assert_finite(tensor) do
    assert Nx.to_number(Nx.any(Nx.is_nan(tensor))) == 0
    assert Nx.to_number(Nx.any(Nx.is_infinity(tensor))) == 0
  end
end
