defmodule HfPeftEx do
  @moduledoc """
  Elixir port of HuggingFace's PEFT (Parameter-Efficient Fine-Tuning) library.

  PEFT methods enable efficient adaptation of large pretrained models by only
  fine-tuning a small number of parameters instead of the entire model.

  ## Supported PEFT Types

  - `:lora` - Low-Rank Adaptation
  - `:adalora` - Adaptive LoRA with rank allocation
  - `:ia3` - Infused Adapter by Inhibiting and Amplifying Inner Activations
  - `:prefix_tuning` - Prepending trainable prefixes
  - `:prompt_tuning` - Learning soft prompts
  - And 25+ more methods

  ## Example

      config = HfPeftEx.Tuners.Lora.Config.new(
        r: 16,
        lora_alpha: 32,
        target_modules: ["q_proj", "v_proj"]
      )

  """

  @version "0.1.0"

  @doc """
  Returns the current version of HfPeftEx.
  """
  @spec version() :: String.t()
  def version, do: @version
end
