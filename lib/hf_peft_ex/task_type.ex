defmodule HfPeftEx.TaskType do
  @moduledoc """
  Enum for the different types of tasks supported by PEFT.

  These task types correspond to common NLP and ML tasks that PEFT methods
  can be applied to.

  ## Supported Task Types

  - `:seq_cls` - Sequence/Text Classification
  - `:seq_2_seq_lm` - Sequence-to-Sequence Language Modeling
  - `:causal_lm` - Causal/Autoregressive Language Modeling
  - `:token_cls` - Token Classification (NER, POS tagging)
  - `:question_ans` - Question Answering
  - `:feature_extraction` - Feature/Embedding Extraction

  """

  @type t ::
          :seq_cls
          | :seq_2_seq_lm
          | :causal_lm
          | :token_cls
          | :question_ans
          | :feature_extraction

  @task_types [
    :seq_cls,
    :seq_2_seq_lm,
    :causal_lm,
    :token_cls,
    :question_ans,
    :feature_extraction
  ]

  @doc """
  Returns all supported task types.

  ## Examples

      iex> HfPeftEx.TaskType.all()
      [:seq_cls, :seq_2_seq_lm, :causal_lm, :token_cls, :question_ans, :feature_extraction]

  """
  @spec all() :: [t()]
  def all, do: @task_types

  @doc """
  Validates if the given atom is a valid task type.

  ## Examples

      iex> HfPeftEx.TaskType.valid?(:causal_lm)
      true

      iex> HfPeftEx.TaskType.valid?(:invalid)
      false

  """
  @spec valid?(atom()) :: boolean()
  def valid?(type) when is_atom(type), do: type in @task_types
  def valid?(_), do: false

  @doc """
  Converts a string to a task type atom.

  ## Examples

      iex> HfPeftEx.TaskType.from_string("CAUSAL_LM")
      {:ok, :causal_lm}

      iex> HfPeftEx.TaskType.from_string("invalid")
      {:error, "Unknown task type: invalid"}

  """
  @spec from_string(String.t()) :: {:ok, t()} | {:error, String.t()}
  def from_string(str) when is_binary(str) do
    type = str |> String.downcase() |> String.to_atom()

    if valid?(type) do
      {:ok, type}
    else
      {:error, "Unknown task type: #{str}"}
    end
  end
end
