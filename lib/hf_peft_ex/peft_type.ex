defmodule HfPeftEx.PeftType do
  @moduledoc """
  Enum for the different types of adapters in PEFT.

  Each PEFT type represents a different parameter-efficient fine-tuning method.

  ## Supported Types

  - `:lora` - Low-Rank Adaptation
  - `:adalora` - Adaptive LoRA
  - `:ia3` - Infused Adapter by Inhibiting and Amplifying
  - `:prefix_tuning` - Prefix Tuning
  - `:prompt_tuning` - Prompt Tuning
  - `:p_tuning` - P-Tuning
  - `:loha` - Low-Rank Hadamard Product
  - `:lokr` - Low-Rank Kronecker Product
  - `:oft` - Orthogonal Fine-Tuning
  - `:boft` - Butterfly Orthogonal Fine-Tuning
  - And more...

  """

  @type t ::
          :prompt_tuning
          | :multitask_prompt_tuning
          | :p_tuning
          | :prefix_tuning
          | :lora
          | :adalora
          | :boft
          | :adaption_prompt
          | :ia3
          | :loha
          | :lokr
          | :oft
          | :poly
          | :ln_tuning
          | :vera
          | :fourierft
          | :xlora
          | :hra
          | :vblora
          | :cpt
          | :bone
          | :miss
          | :randlora
          | :road
          | :trainable_tokens
          | :shira
          | :c3a
          | :waveft
          | :osf
          | :delora
          | :gralora

  @peft_types [
    :prompt_tuning,
    :multitask_prompt_tuning,
    :p_tuning,
    :prefix_tuning,
    :lora,
    :adalora,
    :boft,
    :adaption_prompt,
    :ia3,
    :loha,
    :lokr,
    :oft,
    :poly,
    :ln_tuning,
    :vera,
    :fourierft,
    :xlora,
    :hra,
    :vblora,
    :cpt,
    :bone,
    :miss,
    :randlora,
    :road,
    :trainable_tokens,
    :shira,
    :c3a,
    :waveft,
    :osf,
    :delora,
    :gralora
  ]

  @doc """
  Returns all supported PEFT types.

  ## Examples

      iex> HfPeftEx.PeftType.all()
      [:prompt_tuning, :multitask_prompt_tuning, :p_tuning, ...]

  """
  @spec all() :: [t()]
  def all, do: @peft_types

  @doc """
  Validates if the given atom is a valid PEFT type.

  ## Examples

      iex> HfPeftEx.PeftType.valid?(:lora)
      true

      iex> HfPeftEx.PeftType.valid?(:invalid)
      false

  """
  @spec valid?(atom()) :: boolean()
  def valid?(type) when is_atom(type), do: type in @peft_types
  def valid?(_), do: false

  @doc """
  Converts a string to a PEFT type atom.

  ## Examples

      iex> HfPeftEx.PeftType.from_string("LORA")
      {:ok, :lora}

      iex> HfPeftEx.PeftType.from_string("invalid")
      {:error, "Unknown PEFT type: invalid"}

  """
  @spec from_string(String.t()) :: {:ok, t()} | {:error, String.t()}
  def from_string(str) when is_binary(str) do
    type = str |> String.downcase() |> String.to_atom()

    if valid?(type) do
      {:ok, type}
    else
      {:error, "Unknown PEFT type: #{str}"}
    end
  end

  @doc """
  Checks if the given PEFT type is a prompt learning method.

  Prompt learning methods include prompt tuning, prefix tuning, and P-tuning.

  ## Examples

      iex> HfPeftEx.PeftType.prompt_learning?(:prefix_tuning)
      true

      iex> HfPeftEx.PeftType.prompt_learning?(:lora)
      false

  """
  @spec prompt_learning?(t()) :: boolean()
  def prompt_learning?(type)
      when type in [:prompt_tuning, :multitask_prompt_tuning, :p_tuning, :prefix_tuning, :cpt],
      do: true

  def prompt_learning?(_), do: false
end
