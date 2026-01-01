defmodule HfPeftEx.Tuners.Adalora.RankAllocatorTest do
  use ExUnit.Case, async: true

  alias HfPeftEx.Tuners.Adalora.{Config, RankAllocator}

  describe "new/2" do
    test "creates allocator with config" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config)

      assert allocator.config == config
      assert allocator.ipt == %{}
      assert allocator.exp_avg_ipt == %{}
      assert allocator.exp_avg_unc == %{}
    end

    test "uses custom adapter name" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config, "custom_adapter")

      assert allocator.adapter_name == "custom_adapter"
    end
  end

  describe "compute_importance/2" do
    test "computes |param * gradient|" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config)

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[1.0], [2.0], [3.0]]),
          lora_e_grad: Nx.tensor([[0.1], [0.2], [0.3]])
        }
      }

      updated = RankAllocator.compute_importance(allocator, gradients)

      expected = Nx.tensor([[0.1], [0.4], [0.9]])
      assert_all_close(updated.ipt["layer1"], expected)
    end

    test "handles multiple layers" do
      config = Config.new(total_step: 1000)
      allocator = RankAllocator.new(config)

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[1.0], [2.0]]),
          lora_e_grad: Nx.tensor([[0.5], [0.5]])
        },
        "layer2" => %{
          lora_e: Nx.tensor([[3.0], [4.0]]),
          lora_e_grad: Nx.tensor([[0.1], [0.1]])
        }
      }

      updated = RankAllocator.compute_importance(allocator, gradients)

      assert_all_close(updated.ipt["layer1"], Nx.tensor([[0.5], [1.0]]))
      assert_all_close(updated.ipt["layer2"], Nx.tensor([[0.3], [0.4]]))
    end
  end

  describe "update_ema/1" do
    test "initializes EMA on first update" do
      config = Config.new(total_step: 1000, beta1: 0.5, beta2: 0.5)
      allocator = RankAllocator.new(config)

      # First importance
      allocator = %{
        allocator
        | ipt: %{"layer1" => Nx.tensor([[1.0], [2.0]])}
      }

      allocator = RankAllocator.update_ema(allocator)

      # EMA = (1-0.5) * [1, 2] = [0.5, 1.0] (first update with 0 initial)
      expected = Nx.tensor([[0.5], [1.0]])
      assert_all_close(allocator.exp_avg_ipt["layer1"], expected)
    end

    test "applies exponential moving average" do
      config = Config.new(total_step: 1000, beta1: 0.5, beta2: 0.5)

      allocator = %RankAllocator{
        config: config,
        adapter_name: "default",
        ipt: %{"layer1" => Nx.tensor([[1.0], [2.0]])},
        exp_avg_ipt: %{"layer1" => Nx.tensor([[0.0], [0.0]])},
        exp_avg_unc: %{"layer1" => Nx.tensor([[0.0], [0.0]])}
      }

      allocator = RankAllocator.update_ema(allocator)

      # EMA = 0.5 * [0, 0] + 0.5 * [1, 2] = [0.5, 1.0]
      expected = Nx.tensor([[0.5], [1.0]])
      assert_all_close(allocator.exp_avg_ipt["layer1"], expected)

      # Second importance
      allocator = %{
        allocator
        | ipt: %{"layer1" => Nx.tensor([[2.0], [4.0]])}
      }

      allocator = RankAllocator.update_ema(allocator)

      # EMA = 0.5 * [0.5, 1.0] + 0.5 * [2, 4] = [1.25, 2.5]
      expected = Nx.tensor([[1.25], [2.5]])
      assert_all_close(allocator.exp_avg_ipt["layer1"], expected)
    end

    test "tracks uncertainty as deviation from average" do
      config = Config.new(total_step: 1000, beta1: 0.5, beta2: 0.5)

      allocator = %RankAllocator{
        config: config,
        adapter_name: "default",
        ipt: %{"layer1" => Nx.tensor([[1.0], [2.0]])},
        exp_avg_ipt: %{"layer1" => Nx.tensor([[1.0], [2.0]])},
        exp_avg_unc: %{"layer1" => Nx.tensor([[0.0], [0.0]])}
      }

      allocator = RankAllocator.update_ema(allocator)

      # Uncertainty = |ipt - prev_avg| = |[1,2] - [1,2]| = [0, 0]
      # exp_avg_unc = 0.5 * [0, 0] + 0.5 * [0, 0] = [0, 0]
      expected_unc = Nx.tensor([[0.0], [0.0]])
      assert_all_close(allocator.exp_avg_unc["layer1"], expected_unc)
    end
  end

  describe "get_budget/2" do
    test "returns init_r during warmup" do
      config = Config.new(total_step: 1000, tinit: 100, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      assert RankAllocator.get_budget(allocator, 50) == 12.0
      assert RankAllocator.get_budget(allocator, 100) == 12.0
    end

    test "uses cubic schedule for budget decrease" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      # At step 500 (50% progress)
      # progress = 500/1000 = 0.5
      # budget = (init - target) * (1 - progress)^3 + target
      # budget = 4 * 0.125 + 8 = 8.5
      budget = RankAllocator.get_budget(allocator, 500)
      assert_in_delta budget, 8.5, 0.1
    end

    test "returns target_r at end" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      budget = RankAllocator.get_budget(allocator, 1000)
      assert budget == 8.0
    end

    test "accounts for tinit and tfinal in schedule" do
      config = Config.new(total_step: 1000, tinit: 100, tfinal: 100, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      # At step 100 (end of warmup), budget should still be near init
      budget = RankAllocator.get_budget(allocator, 100)
      assert budget == 12.0

      # At step 900 (start of final), should be at target
      budget = RankAllocator.get_budget(allocator, 900)
      assert budget == 8.0
    end
  end

  describe "compute_masks/2" do
    test "creates masks based on importance threshold" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)

      allocator = %RankAllocator{
        config: config,
        adapter_name: "default",
        ipt: %{},
        exp_avg_ipt: %{
          "layer1" => Nx.tensor([[1.0], [0.5], [0.2], [0.1]]),
          "layer2" => Nx.tensor([[0.8], [0.3], [0.15], [0.05]])
        },
        exp_avg_unc: %{
          "layer1" => Nx.tensor([[0.0], [0.0], [0.0], [0.0]]),
          "layer2" => Nx.tensor([[0.0], [0.0], [0.0], [0.0]])
        }
      }

      # Budget = 2 per layer (target_r)
      masks = RankAllocator.compute_masks(allocator, 2.0)

      # Should keep top 50% by importance (2 out of 4 per layer)
      assert Nx.to_number(Nx.sum(masks["layer1"])) == 2
      assert Nx.to_number(Nx.sum(masks["layer2"])) == 2
    end

    test "includes uncertainty in scoring" do
      config = Config.new(total_step: 1000, init_r: 4, target_r: 2)

      allocator = %RankAllocator{
        config: config,
        adapter_name: "default",
        ipt: %{},
        exp_avg_ipt: %{
          "layer1" => Nx.tensor([[0.5], [0.5], [0.5], [0.5]])
        },
        exp_avg_unc: %{
          # Uncertainty makes first two more important
          "layer1" => Nx.tensor([[0.5], [0.4], [0.0], [0.0]])
        }
      }

      masks = RankAllocator.compute_masks(allocator, 2.0)

      # First two should be kept due to higher uncertainty
      mask_list = Nx.to_flat_list(masks["layer1"])
      assert Enum.at(mask_list, 0) == 1
      assert Enum.at(mask_list, 1) == 1
    end
  end

  describe "update_and_allocate/3" do
    test "returns nil during warmup" do
      config = Config.new(total_step: 1000, tinit: 100)
      allocator = RankAllocator.new(config)

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, %{}, 50)
      assert masks == nil
    end

    test "returns nil during final phase" do
      config = Config.new(total_step: 1000, tfinal: 100)
      allocator = RankAllocator.new(config)

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, %{}, 950)
      assert masks == nil
    end

    test "returns nil when not at pruning interval" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10)
      allocator = RankAllocator.new(config)

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, %{}, 5)
      assert masks == nil
    end

    test "returns masks at pruning intervals" do
      config =
        Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10, init_r: 12, target_r: 4)

      allocator = %RankAllocator{
        config: config,
        adapter_name: "default",
        ipt: %{},
        exp_avg_ipt: %{"layer1" => Nx.tensor([[1.0], [0.5]])},
        exp_avg_unc: %{"layer1" => Nx.tensor([[0.0], [0.0]])}
      }

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[1.0], [1.0]]),
          lora_e_grad: Nx.tensor([[0.1], [0.2]])
        }
      }

      {_allocator, masks} = RankAllocator.update_and_allocate(allocator, gradients, 100)
      assert masks != nil
    end

    test "updates importance and EMA before computing masks" do
      config =
        Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10, init_r: 12, target_r: 4)

      allocator = RankAllocator.new(config)

      gradients = %{
        "layer1" => %{
          lora_e: Nx.tensor([[2.0], [1.0]]),
          lora_e_grad: Nx.tensor([[0.5], [0.5]])
        }
      }

      {updated_allocator, _masks} =
        RankAllocator.update_and_allocate(allocator, gradients, 10)

      # Should have computed importance
      assert Map.has_key?(updated_allocator.ipt, "layer1")
      # Should have updated EMA
      assert Map.has_key?(updated_allocator.exp_avg_ipt, "layer1")
    end
  end

  describe "schedule_should_mask?/2" do
    test "returns false during warmup" do
      config = Config.new(total_step: 1000, tinit: 100, delta_t: 10)
      allocator = RankAllocator.new(config)

      refute RankAllocator.schedule_should_mask?(allocator, 50)
      refute RankAllocator.schedule_should_mask?(allocator, 100)
    end

    test "returns false during final phase" do
      config = Config.new(total_step: 1000, tfinal: 100, delta_t: 10)
      allocator = RankAllocator.new(config)

      refute RankAllocator.schedule_should_mask?(allocator, 901)
      refute RankAllocator.schedule_should_mask?(allocator, 950)
    end

    test "returns true at delta_t intervals" do
      config = Config.new(total_step: 1000, tinit: 0, tfinal: 0, delta_t: 10)
      allocator = RankAllocator.new(config)

      assert RankAllocator.schedule_should_mask?(allocator, 10)
      assert RankAllocator.schedule_should_mask?(allocator, 100)
      refute RankAllocator.schedule_should_mask?(allocator, 15)
    end
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    assert Nx.all_close(a, b, atol: atol) |> Nx.to_number() == 1
  end
end
