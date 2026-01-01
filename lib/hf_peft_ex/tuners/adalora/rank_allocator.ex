defmodule HfPeftEx.Tuners.Adalora.RankAllocator do
  @moduledoc """
  Manages adaptive rank allocation during AdaLoRA training.

  The RankAllocator tracks importance scores for each singular value across
  all AdaLoRA layers and determines which values to prune based on a
  decreasing budget schedule.

  ## Budget Schedule

  The rank budget decreases from `init_r` to `target_r` following a cubic
  schedule during the budgeting phase (between `tinit` and `total_step - tfinal`).

  ## Importance Scoring

  Importance is computed as `|lambda * grad(lambda)|` where `lambda` are the
  singular values (lora_e). An exponential moving average (EMA) is used for
  smoothing, along with uncertainty quantification for exploration.

  ## Example

      config = Config.new(total_step: 10000, init_r: 12, target_r: 8)
      allocator = RankAllocator.new(config)

      # In training loop, after backward pass
      gradients = collect_gradients(model)
      {allocator, masks} = RankAllocator.update_and_allocate(allocator, gradients, step)

      if masks do
        apply_masks_to_model(model, masks)
      end

  """

  alias HfPeftEx.Tuners.Adalora.Config

  @type t :: %__MODULE__{
          config: Config.t(),
          adapter_name: String.t(),
          ipt: %{String.t() => Nx.Tensor.t()},
          exp_avg_ipt: %{String.t() => Nx.Tensor.t()},
          exp_avg_unc: %{String.t() => Nx.Tensor.t()}
        }

  defstruct config: nil,
            adapter_name: "default",
            ipt: %{},
            exp_avg_ipt: %{},
            exp_avg_unc: %{}

  @doc """
  Creates a new RankAllocator.

  ## Arguments

    * `config` - AdaLoRA configuration
    * `adapter_name` - Name of the adapter (default: `"default"`)

  """
  @spec new(Config.t(), String.t()) :: t()
  def new(%Config{} = config, adapter_name \\ "default") do
    %__MODULE__{
      config: config,
      adapter_name: adapter_name
    }
  end

  @doc """
  Computes importance scores from parameter gradients.

  Importance is computed as `|param * gradient|` for each singular value.

  ## Arguments

    * `allocator` - The RankAllocator
    * `gradients` - Map of layer name to `%{lora_e: tensor, lora_e_grad: tensor}`

  """
  @spec compute_importance(t(), map()) :: t()
  def compute_importance(%__MODULE__{} = allocator, gradients) do
    new_ipt =
      for {name, %{lora_e: e, lora_e_grad: grad}} <- gradients, into: %{} do
        # Importance = |param * gradient|
        ipt = Nx.abs(Nx.multiply(e, grad))
        {name, ipt}
      end

    %{allocator | ipt: new_ipt}
  end

  @doc """
  Updates exponential moving averages for importance and uncertainty.

  Uses `beta1` for importance smoothing and `beta2` for uncertainty.
  """
  @spec update_ema(t()) :: t()
  def update_ema(%__MODULE__{} = allocator) do
    beta1 = allocator.config.beta1
    beta2 = allocator.config.beta2

    new_exp_avg_ipt =
      for {name, ipt} <- allocator.ipt, into: %{} do
        prev = Map.get(allocator.exp_avg_ipt, name, Nx.broadcast(0.0, Nx.shape(ipt)))
        new_avg = Nx.add(Nx.multiply(prev, beta1), Nx.multiply(ipt, 1 - beta1))
        {name, new_avg}
      end

    new_exp_avg_unc =
      for {name, ipt} <- allocator.ipt, into: %{} do
        prev_avg = Map.get(allocator.exp_avg_ipt, name, ipt)
        prev_unc = Map.get(allocator.exp_avg_unc, name, Nx.broadcast(0.0, Nx.shape(ipt)))
        # Uncertainty as |current - average|
        unc = Nx.abs(Nx.subtract(ipt, prev_avg))
        new_unc = Nx.add(Nx.multiply(prev_unc, beta2), Nx.multiply(unc, 1 - beta2))
        {name, new_unc}
      end

    %{allocator | exp_avg_ipt: new_exp_avg_ipt, exp_avg_unc: new_exp_avg_unc}
  end

  @doc """
  Gets the current budget based on training step.

  Uses a cubic schedule to decrease from `init_r` to `target_r`.
  """
  @spec get_budget(t(), non_neg_integer()) :: float()
  def get_budget(%__MODULE__{} = allocator, global_step) do
    config = allocator.config
    tinit = config.tinit
    tfinal = config.tfinal
    total = config.total_step

    cond do
      global_step <= tinit ->
        config.init_r * 1.0

      global_step >= total - tfinal ->
        config.target_r * 1.0

      true ->
        # Cubic schedule for smooth budget decrease
        progress = (global_step - tinit) / (total - tfinal - tinit)
        # (1 - progress)^3 gives a cubic decrease curve
        mul_coeff = :math.pow(1 - progress, 3)
        (config.init_r - config.target_r) * mul_coeff + config.target_r
    end
  end

  @doc """
  Computes masks for each layer to meet the budget.

  The combined score (importance + uncertainty) is used to rank singular
  values globally, and a threshold is computed to achieve the target budget.
  """
  @spec compute_masks(t(), float()) :: %{String.t() => Nx.Tensor.t()}
  def compute_masks(%__MODULE__{} = allocator, budget) do
    # Collect all importance scores with layer names
    all_scores =
      for {name, ipt} <- allocator.exp_avg_ipt do
        unc = Map.get(allocator.exp_avg_unc, name, ipt)
        # Combined score (importance * uncertainty for exploration)
        combined = Nx.multiply(ipt, unc)
        {name, combined, ipt}
      end

    # Count total parameters across all layers
    num_layers = map_size(allocator.exp_avg_ipt)
    total_budget = trunc(budget * num_layers)

    # Flatten and find threshold
    all_importance =
      Enum.flat_map(all_scores, fn {_, _, ipt} ->
        Nx.to_flat_list(ipt)
      end)

    total_params = length(all_importance)
    keep_count = min(total_budget, total_params)

    threshold =
      if keep_count >= total_params do
        0.0
      else
        sorted = Enum.sort(all_importance, :desc)
        Enum.at(sorted, max(keep_count - 1, 0), 0.0)
      end

    # Generate masks - keep values >= threshold
    for {name, _combined, ipt} <- all_scores, into: %{} do
      mask =
        ipt
        |> Nx.greater_equal(threshold)
        |> Nx.as_type(:f32)

      {name, mask}
    end
  end

  @doc """
  Checks if masking should occur at this step based on the schedule.
  """
  @spec schedule_should_mask?(t(), non_neg_integer()) :: boolean()
  def schedule_should_mask?(%__MODULE__{} = allocator, global_step) do
    config = allocator.config

    cond do
      # Warmup phase: no pruning
      global_step <= config.tinit ->
        false

      # Final phase: no more pruning
      global_step > config.total_step - config.tfinal ->
        false

      # During budgeting: prune at delta_t intervals
      rem(global_step, config.delta_t) == 0 ->
        true

      true ->
        false
    end
  end

  @doc """
  Main update function called each training step.

  Returns `{updated_allocator, masks}` where masks is nil if no pruning
  occurs at this step, or a map of layer name to mask tensors.
  """
  @spec update_and_allocate(t(), map(), non_neg_integer()) ::
          {t(), %{String.t() => Nx.Tensor.t()} | nil}
  def update_and_allocate(%__MODULE__{} = allocator, gradients, global_step) do
    config = allocator.config

    cond do
      # Warmup phase: no pruning
      global_step <= config.tinit ->
        {allocator, nil}

      # Final phase: freeze ranks
      global_step > config.total_step - config.tfinal ->
        {allocator, nil}

      # Pruning interval
      rem(global_step, config.delta_t) == 0 ->
        allocator =
          allocator
          |> compute_importance(gradients)
          |> update_ema()

        budget = get_budget(allocator, global_step)
        masks = compute_masks(allocator, budget)

        {allocator, masks}

      true ->
        {allocator, nil}
    end
  end

  @doc """
  Resets all importance tracking state.

  Called when transitioning to the final phase.
  """
  @spec reset_ipt(t()) :: t()
  def reset_ipt(%__MODULE__{} = allocator) do
    %{allocator | ipt: %{}, exp_avg_ipt: %{}, exp_avg_unc: %{}}
  end
end
