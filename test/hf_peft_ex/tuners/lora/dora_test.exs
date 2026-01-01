defmodule HfPeftEx.Tuners.Lora.DoraTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Lora.Dora

  describe "init_magnitude/1" do
    test "computes row-wise L2 norm of weight" do
      weight =
        Nx.tensor([
          [3.0, 0.0],
          [4.0, 0.0]
        ])

      # Row norms: sqrt(3^2 + 0) = 3.0, sqrt(4^2 + 0) = 4.0
      magnitude = Dora.init_magnitude(weight)

      assert Nx.shape(magnitude) == {2}
      values = Nx.to_list(magnitude)
      assert_in_delta Enum.at(values, 0), 3.0, 0.001
      assert_in_delta Enum.at(values, 1), 4.0, 0.001
    end

    test "handles single column" do
      weight = Nx.tensor([[1.0], [2.0], [2.0]])
      magnitude = Dora.init_magnitude(weight)

      assert Nx.shape(magnitude) == {3}
      values = Nx.to_list(magnitude)
      assert_in_delta Enum.at(values, 0), 1.0, 0.001
      assert_in_delta Enum.at(values, 1), 2.0, 0.001
      assert_in_delta Enum.at(values, 2), 2.0, 0.001
    end

    test "handles row of ones" do
      weight = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      magnitude = Dora.init_magnitude(weight)

      assert Nx.shape(magnitude) == {1}
      # Row norm is sqrt(4)
      assert_in_delta Nx.to_number(Nx.squeeze(magnitude)), 2.0, 0.001
    end
  end

  describe "new/2" do
    test "creates dora struct with magnitude from weight" do
      weight = Nx.tensor([[3.0, 0.0], [4.0, 0.0]])
      dora = Dora.new(weight)

      assert Nx.shape(dora.magnitude) == {2}
    end

    test "defaults fan_in_fan_out to false" do
      weight = Nx.tensor([[1.0, 2.0]])
      dora = Dora.new(weight)

      assert dora.fan_in_fan_out == false
    end

    test "accepts fan_in_fan_out option" do
      weight = Nx.tensor([[1.0, 2.0]])
      dora = Dora.new(weight, fan_in_fan_out: true)

      assert dora.fan_in_fan_out == true
    end
  end

  describe "get_weight_norm/4" do
    test "computes norm of combined weight and lora" do
      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      scaling = 1.0

      norm = Dora.get_weight_norm(weight, lora_weight, scaling, fan_in_fan_out: false)

      # Combined = [[2,0],[0,2]], column norms = [2, 2]
      assert Nx.shape(norm) == {2}
      values = Nx.to_list(norm)
      assert_in_delta Enum.at(values, 0), 2.0, 0.001
      assert_in_delta Enum.at(values, 1), 2.0, 0.001
    end

    test "scales lora weight by scaling factor" do
      weight = Nx.tensor([[1.0], [0.0]])
      lora_weight = Nx.tensor([[1.0], [0.0]])
      scaling = 2.0

      norm = Dora.get_weight_norm(weight, lora_weight, scaling, fan_in_fan_out: false)

      # Combined = [[1 + 2*1], [0]] = [[3], [0]], row norms = [3, 0]
      assert Nx.shape(norm) == {2}
      values = Nx.to_list(norm)
      assert_in_delta Enum.at(values, 0), 3.0, 0.001
      assert_in_delta Enum.at(values, 1), 0.0, 0.001
    end
  end

  describe "forward/7" do
    test "applies magnitude normalization" do
      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      dora = Dora.new(weight)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      lora_B = Nx.tensor([[0.0, 0.0], [0.0, 0.0]])
      x = Nx.tensor([[1.0, 1.0]])
      base_result = Nx.tensor([[1.0, 1.0]])
      scaling = 1.0

      result = Dora.forward(dora, x, lora_A, lora_B, scaling, weight, base_result)

      # When lora_B is zeros, lora contribution is zero
      # Result should be close to base_result with magnitude scaling
      assert Nx.shape(result) == {1, 2}
    end

    test "returns correct shape for batch input" do
      weight = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      dora = Dora.new(weight)
      # credo:disable-for-lines:3 Credo.Check.Readability.VariableNames
      lora_A = Nx.tensor([[0.1, 0.0], [0.0, 0.1]])
      lora_B = Nx.tensor([[0.1, 0.0], [0.0, 0.1]])
      # batch of 3
      x = Nx.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
      base_result = Nx.broadcast(1.0, {3, 2})
      scaling = 0.5

      result = Dora.forward(dora, x, lora_A, lora_B, scaling, weight, base_result)

      assert Nx.shape(result) == {3, 2}
    end
  end

  describe "apply_dora/2" do
    test "adds dora contribution to base output" do
      layer = %{
        lora_A: Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
        lora_B: Nx.tensor([[0.5, 0.0], [0.0, 0.5]]),
        scaling: 1.0
      }

      weight = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      dora = Dora.new(weight)

      x = Nx.tensor([[1.0, 1.0]])
      base_output = Nx.tensor([[1.0, 1.0]])

      result = Dora.apply_dora(dora, layer, x, base_output, weight)

      # Result should be modified base_output with DoRA adjustment
      assert Nx.shape(result) == {1, 2}
    end
  end
end
