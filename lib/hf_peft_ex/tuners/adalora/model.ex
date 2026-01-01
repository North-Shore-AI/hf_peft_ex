defmodule HfPeftEx.Tuners.Adalora.Model do
  @moduledoc """
  AdaLoRA model that wraps base layers with adaptive rank allocation.

  The Model manages multiple AdaLoRA layers and coordinates rank allocation
  during training. It provides methods for:

  - Forward pass through all wrapped layers
  - Rank allocation updates during training
  - Orthogonal regularization loss computation
  - Weight merging/unmerging for inference
  - Resizing layers based on final rank pattern

  ## Example

      config = Config.new(
        total_step: 10000,
        init_r: 12,
        target_r: 8,
        target_modules: ["q_proj", "v_proj"]
      )

      base_layers = %{
        "q_proj" => %{weight: q_weight, bias: nil},
        "v_proj" => %{weight: v_weight, bias: nil}
      }

      model = Model.new(config, base_layers)

      # In training loop
      outputs = Model.forward(model, inputs)
      loss = compute_loss(outputs, targets)

      # Add orthogonal regularization
      orth_loss = Model.get_orthogonal_loss(model)
      total_loss = loss + config.orth_reg_weight * orth_loss

      # After backward pass
      gradients = collect_gradients(model)
      {model, masks} = Model.update_and_allocate(model, gradients, step)

  """

  alias HfPeftEx.Tuners.Adalora.{Config, Layer, RankAllocator}

  @type t :: %__MODULE__{
          config: Config.t(),
          adapter_name: String.t(),
          layers: %{String.t() => Layer.t()},
          rank_allocator: RankAllocator.t()
        }

  defstruct config: nil,
            adapter_name: "default",
            layers: %{},
            rank_allocator: nil

  @doc """
  Creates a new AdaLoRA model.

  ## Arguments

    * `config` - AdaLoRA configuration
    * `base_layers` - Map of layer name to base layer maps with `:weight` and `:bias`
    * `opts` - Options

  ## Options

    * `:adapter_name` - Name for the adapter (default: `"default"`)

  """
  @spec new(Config.t(), map(), keyword()) :: t()
  def new(%Config{} = config, base_layers, opts \\ []) do
    adapter_name = Keyword.get(opts, :adapter_name, "default")

    layers =
      for {name, base_layer} <- base_layers, into: %{} do
        layer =
          base_layer
          |> Layer.new()
          |> Layer.update_layer(adapter_name, config.init_r, config.lora_alpha)

        {name, layer}
      end

    rank_allocator = RankAllocator.new(config, adapter_name)

    %__MODULE__{
      config: config,
      adapter_name: adapter_name,
      layers: layers,
      rank_allocator: rank_allocator
    }
  end

  @doc """
  Forward pass through all layers.

  ## Arguments

    * `model` - The AdaLoRA model
    * `inputs` - Map of layer name to input tensor

  ## Returns

  Map of layer name to output tensor.
  """
  @spec forward(t(), map()) :: map()
  def forward(%__MODULE__{} = model, inputs) do
    for {name, input} <- inputs, into: %{} do
      case Map.get(model.layers, name) do
        nil ->
          {name, input}

        layer ->
          output = Layer.forward(layer, input)
          {name, output}
      end
    end
  end

  @doc """
  Updates importance scores and allocates rank budget.

  Should be called after backward pass and before optimizer step.

  ## Arguments

    * `model` - The AdaLoRA model
    * `gradients` - Map of layer name to gradient info
    * `global_step` - Current training step

  ## Returns

  `{updated_model, masks}` where masks is nil if no pruning occurs.
  """
  @spec update_and_allocate(t(), map(), non_neg_integer()) ::
          {t(), %{String.t() => Nx.Tensor.t()} | nil}
  def update_and_allocate(%__MODULE__{} = model, gradients, global_step) do
    {updated_allocator, masks} =
      RankAllocator.update_and_allocate(model.rank_allocator, gradients, global_step)

    model = %{model | rank_allocator: updated_allocator}

    # Apply masks to layers if pruning occurred
    model =
      if masks do
        apply_masks(model, masks)
      else
        model
      end

    {model, masks}
  end

  defp apply_masks(model, masks) do
    updated_layers =
      for {name, layer} <- model.layers, into: %{} do
        case Map.get(masks, name) do
          nil ->
            {name, layer}

          mask ->
            updated_layer = Layer.apply_mask(layer, model.adapter_name, mask)
            {name, updated_layer}
        end
      end

    %{model | layers: updated_layers}
  end

  @doc """
  Computes orthogonal regularization loss.

  Encourages A and B matrices to maintain orthogonality:
  `loss = sum(||A @ A^T - I||_F + ||B^T @ B - I||_F) / num_layers`
  """
  @spec get_orthogonal_loss(t()) :: Nx.Tensor.t()
  def get_orthogonal_loss(%__MODULE__{layers: layers}) when map_size(layers) == 0 do
    Nx.tensor(0.0)
  end

  def get_orthogonal_loss(%__MODULE__{} = model) do
    losses =
      for {_name, layer} <- model.layers do
        case Map.get(layer.lora_a, model.adapter_name) do
          nil ->
            Nx.tensor(0.0)

          lora_a ->
            lora_b = layer.lora_b[model.adapter_name]

            # A @ A^T should be identity (r x r)
            a_aat = Nx.dot(lora_a, Nx.transpose(lora_a))
            identity_a = Nx.eye(Nx.axis_size(a_aat, 0))
            loss_a = Nx.LinAlg.norm(Nx.subtract(a_aat, identity_a))

            # B^T @ B should be identity (r x r)
            bt_b = Nx.dot(Nx.transpose(lora_b), lora_b)
            identity_b = Nx.eye(Nx.axis_size(bt_b, 0))
            loss_b = Nx.LinAlg.norm(Nx.subtract(bt_b, identity_b))

            Nx.add(loss_a, loss_b)
        end
      end

    if Enum.empty?(losses) do
      Nx.tensor(0.0)
    else
      losses
      |> Nx.stack()
      |> Nx.mean()
    end
  end

  @doc """
  Merges LoRA weights into base weights.

  Returns `{merged_model, merged_weights}`.
  """
  @spec merge(t()) :: {t(), %{String.t() => Nx.Tensor.t()}}
  def merge(%__MODULE__{} = model) do
    {updated_layers, merged_weights} =
      Enum.reduce(model.layers, {%{}, %{}}, fn {name, layer}, {layers_acc, weights_acc} ->
        base_weight = layer.base_layer.weight
        {merged_layer, new_weight} = Layer.merge(layer, model.adapter_name, base_weight)
        {Map.put(layers_acc, name, merged_layer), Map.put(weights_acc, name, new_weight)}
      end)

    {%{model | layers: updated_layers}, merged_weights}
  end

  @doc """
  Unmerges LoRA weights from base weights.

  Returns `{unmerged_model, restored_weights}`.
  """
  @spec unmerge(t(), map()) :: {t(), %{String.t() => Nx.Tensor.t()}}
  def unmerge(%__MODULE__{} = model, merged_weights) do
    {updated_layers, restored_weights} =
      Enum.reduce(model.layers, {%{}, %{}}, fn {name, layer}, {layers_acc, weights_acc} ->
        merged_weight = Map.get(merged_weights, name, layer.base_layer.weight)

        {unmerged_layer, restored_weight} =
          Layer.unmerge(layer, model.adapter_name, merged_weight)

        {Map.put(layers_acc, name, unmerged_layer), Map.put(weights_acc, name, restored_weight)}
      end)

    {%{model | layers: updated_layers}, restored_weights}
  end

  @doc """
  Resizes layers based on a rank pattern.

  The rank pattern specifies which singular values to keep for each layer.
  This is used after training to finalize the model with reduced ranks.

  ## Arguments

    * `model` - The AdaLoRA model
    * `rank_pattern` - Map of layer name to list of booleans indicating which ranks to keep

  """
  @spec resize_by_rank_pattern(t(), map()) :: t()
  def resize_by_rank_pattern(%__MODULE__{} = model, rank_pattern) do
    updated_layers =
      for {name, layer} <- model.layers, into: %{} do
        case Map.get(rank_pattern, name) do
          nil ->
            {name, layer}

          pattern ->
            resized_layer = resize_layer(layer, model.adapter_name, pattern)
            {name, resized_layer}
        end
      end

    %{model | layers: updated_layers}
  end

  defp resize_layer(layer, adapter_name, pattern) do
    keep_indices =
      pattern
      |> Enum.with_index()
      |> Enum.filter(fn {keep, _idx} -> keep end)
      |> Enum.map(fn {_keep, idx} -> idx end)

    new_rank = length(keep_indices)

    if new_rank == 0 do
      # All pruned - set everything to zeros with minimal rank
      layer
      |> put_in([Access.key(:lora_a), adapter_name], Nx.broadcast(0.0, {1, layer.in_features}))
      |> put_in([Access.key(:lora_b), adapter_name], Nx.broadcast(0.0, {layer.out_features, 1}))
      |> put_in([Access.key(:lora_e), adapter_name], Nx.broadcast(0.0, {1, 1}))
      |> put_in([Access.key(:ranknum), adapter_name], 0)
    else
      lora_a = layer.lora_a[adapter_name]
      lora_b = layer.lora_b[adapter_name]
      lora_e = layer.lora_e[adapter_name]

      # Select rows/columns based on keep_indices
      indices = Nx.tensor(keep_indices)

      new_lora_a = Nx.take(lora_a, indices, axis: 0)
      new_lora_b = Nx.take(lora_b, indices, axis: 1)
      new_lora_e = Nx.take(lora_e, indices, axis: 0)

      layer
      |> put_in([Access.key(:lora_a), adapter_name], new_lora_a)
      |> put_in([Access.key(:lora_b), adapter_name], new_lora_b)
      |> put_in([Access.key(:lora_e), adapter_name], new_lora_e)
      |> put_in([Access.key(:ranknum), adapter_name], new_rank)
    end
  end

  @doc """
  Gets the current rank pattern from all layers.
  """
  @spec get_rank_pattern(t()) :: map()
  def get_rank_pattern(%__MODULE__{} = model) do
    for {name, layer} <- model.layers, into: %{} do
      case Map.get(layer.lora_e, model.adapter_name) do
        nil ->
          {name, []}

        lora_e ->
          # Non-zero values are kept
          pattern =
            lora_e
            |> Nx.not_equal(0.0)
            |> Nx.to_flat_list()
            |> Enum.map(&(&1 == 1))

          {name, pattern}
      end
    end
  end
end
