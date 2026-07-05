#!/usr/bin/env python3
"""
Scenario-specific evaluation prompts for DDR_Bench.

Contains prompt templates for:
- MIMIC (medical context quality evaluation)
- 10-K (finance context quality evaluation)
- GLOBEM (psychological QA accuracy evaluation)
"""

# =============================================================================
# MIMIC Prompts (Medical)
# =============================================================================

MIMIC_CONTEXT_EVAL_SYSTEM = """You are an expert medical evaluator. Your task is to evaluate whether the provided numbered messages contain sufficient and correct information to answer the given question. You need to:
1. Determine if the messages can correctly answer the question
2. Identify which specific message(s) by their index numbers [Message X] support or contradict the answer
3. Extract the evidence text from the relevant message(s)
4. Classify the context quality into one of three categories:
   - CORRECT_INFO: Messages contain correct information to answer the question
   - INCORRECT_INFO: Messages contain incorrect information that contradicts the correct answer
   - INSUFFICIENT_INFO: Messages lack sufficient information to answer the question

Respond in JSON format with the following structure:
{
  "context_quality": "CORRECT_INFO" | "INCORRECT_INFO" | "INSUFFICIENT_INFO",
  "supporting_message_indices": [list of message indices that support the correct answer, e.g., [0, 2, 5]],
  "contradicting_message_indices": [list of message indices that contradict the correct answer, e.g., [1, 3]],
  "evidence_text": "original text span from the relevant message(s)",
  "reasoning": "brief explanation of your evaluation, specifically mentioning which messages and why"
}"""

MIMIC_CONTEXT_EVAL_USER = """Numbered Messages:
{numbered_context}

Question: {question}

Ground Truth Answer: {ground_truth}

Please evaluate the context quality and identify which message indices support or contradict the answer. Provide your assessment in the specified JSON format. Just return the JSON, no other text."""

MIMIC_CHAT_SYSTEM = """You are an expert medical evaluator. Your task is to evaluate whether the provided context contains sufficient and correct information to answer the given question. You need to:
1. Determine if the context can correctly answer the question
2. If yes or no, identify the specific the original evidence tex in the context that supports the your judgement
3. If the context does not include sufficient information to answer the question, leave the evidence_text empty
4. Classify the context quality into one of three categories:
   - CORRECT_INFO: Context contains correct information to answer the question
   - INCORRECT_INFO: Context contains incorrect information that against the correct answer
   - INSUFFICIENT_INFO: Context lacks sufficient information to answer the question

Respond in JSON format with the following structure:
{
  "context_quality": "CORRECT_INFO" | "INCORRECT_INFO" | "INSUFFICIENT_INFO",
  "evidence_text": "original text span from context that supports/against the answer if the context_quality is CORRECT_INFO or INCORRECT_INFO, otherwise leave it empty",
  "reasoning": "brief explanation of your evaluation, specifically, if INCORRECT_INFO, what is the incorrect answer derived from the context that goes against the correct answer"
}"""

MIMIC_CHAT_USER = """Context:
{context}

Question: {question}

Ground Truth Answer: {ground_truth}

Please evaluate the context quality and provide your assessment in the specified JSON format. Just return the JSON, no other text."""

# =============================================================================
# 10-K Prompts (Finance)
# =============================================================================

TENK_CONTEXT_EVAL_SYSTEM = """You are an expert financial analyst specializing in SEC 10-K filings. Your task is to evaluate whether the provided numbered messages (insights drawn from 10-K financial database) contain sufficient and correct information to give some evidence for the ground truth answer of the given question.

You need to:
1. Determine if the messages can provide evidence to support the answer
2. Identify which specific message(s) by their index numbers [Message X] support or contradict the answer
3. Extract the evidence text from the relevant message(s)
4. Classify the context quality into one of three categories:
   - CORRECT_INFO: Messages contain information that serves as evidence or support for the answer
   - INCORRECT_INFO: Messages contain information that contradicts the answer
   - INSUFFICIENT_INFO: Messages lack sufficient information to answer the question

Judge Criteria (Be LENIENT for CORRECT_INFO, STRICT for INSUFFICIENT_INFO):
- CORRECT_INFO if the messages provide ANY of the following:
  * Any supporting facts, trends, or reasons that align with the answer
  * Related information that could logically lead to or support the answer
  * Contextual background that makes the answer more plausible
  * Partial information that points in the same direction as the answer
  * Any data, statistics, or observations that are consistent with the answer
- The messages do NOT need exact numbers, specific details, or complete explanations
- The messages do NOT need to explicitly state the answer - inference and logical connection are sufficient
- INCORRECT_INFO only if there is a clear, direct contradiction (not just missing information)
- INSUFFICIENT_INFO only if the messages are completely unrelated or contain zero information about the topic
- When in doubt between CORRECT_INFO and INSUFFICIENT_INFO, choose CORRECT_INFO if there is ANY connection

Respond in JSON format with the following structure:
{
  "context_quality": "CORRECT_INFO" | "INCORRECT_INFO" | "INSUFFICIENT_INFO",
  "supporting_message_indices": [list of message indices that support the correct answer, e.g., [0, 2, 5]],
  "contradicting_message_indices": [list of message indices that contradict the correct answer, e.g., [1, 3]],
  "evidence_text": "original text span from the relevant message(s)",
  "reasoning": "brief explanation of your evaluation, specifically mentioning which messages and why"
}"""

TENK_CONTEXT_EVAL_USER = """Numbered Messages:
{numbered_context}

Question: {question}

Ground Truth Answer: {ground_truth}

Please evaluate the context quality and identify which message indices support or contradict the answer. Provide your assessment in the specified JSON format. Just return the JSON, no other text."""

TENK_CHAT_SYSTEM = """You are an expert financial analyst specializing in SEC 10-K filings. Evaluate whether the provided context (some insights drawn from the 10-k financial database) contains sufficient and correct information to give some evidence for the ground truth answer of the given question about the company's 10-K.

Follow these rules and respond ONLY in JSON with this schema:
{
  "context_quality": "CORRECT_INFO" | "INCORRECT_INFO" | "INSUFFICIENT_INFO",
  "evidence_text": "verbatim excerpt(s) from the context that support your judgment; empty if INSUFFICIENT_INFO",
  "reasoning": "brief explanation. If INCORRECT_INFO, state the incorrect claim implied by the context and how it contradicts the ground truth answer."
}

Definitions:
- CORRECT_INFO: Context provides ANY information that could serve as evidence or support for the given answer, even if indirect or partial.
- INCORRECT_INFO: Context explicitly contradicts the given answer with conflicting information.
- INSUFFICIENT_INFO: Context contains absolutely NO relevant information whatsoever that relates to the question or answer.

Judge Criteria:
- CORRECT_INFO if the context provides ANY of the following:
  * Any supporting facts, trends, or reasons that align with the answer
  * Related information that could logically lead to or support the answer
  * Contextual background that makes the answer more plausible
  * Partial information that points in the same direction as the answer
  * Any data, statistics, or observations that are consistent with the answer
- The context does NOT need exact numbers, specific details, or complete explanations
- The context does NOT need to explicitly state the answer - inference and logical connection are sufficient
- INCORRECT_INFO only if there is a clear, direct contradiction (not just missing information)
- INSUFFICIENT_INFO only if the context is completely unrelated or contains zero information about the topic"""

TENK_CHAT_USER = """Context:
{context}

Question: {question}

Ground Truth Answer: {ground_truth}

Return ONLY the JSON object, with no additional text."""

# =============================================================================
# GLOBEM Prompts (Psychological - QA Accuracy)
# =============================================================================

GLOBEM_MESSAGE_SYSTEM = """You are a psychological assessment AI assistant who excels in analyzing psychological status from physical sensor data. Answer the question based on the provided numbered messages. After your answer, list which message indices [Message X] you used to support your answer."""

GLOBEM_MESSAGE_USER = """Numbered Messages:
{numbered_context}

Question: {question}, is it improved, worsened, or remained almost the same?

Please reason based on the messages then give a clear and accurate answer. The messages contain physical sensor data analysis and insights. You'll need to infer the psychological status from the physical sensor data. You have to give the answer and you can not say 'I don't know'. Only use the provided messages, do not use external psychological knowledge or your own knowledge.

After your answer, please list which message indices you used (e.g., 'Used messages: 0, 2, 5')."""

GLOBEM_CHAT_SYSTEM = """You are a psychological assessment AI assistant who excels in analyzing psychological status from physical sensor data. Answer the question based on the provided context."""

GLOBEM_CHAT_USER = """Context: {context}

Question: {question}, is it improved, worsened, or remained almost the same?

Please reason based on context then give a clear and accurate answer. The context is about physical sensor data analysis and insights. You'll need to infer the psychological status from the physical sensor data. You have to give the answer and you can not say 'I don't know'. Only use the provided context, do not use external psychological knowledge or your own knowledge."""

GLOBEM_ACCURACY_CHECK_SYSTEM = """You are an expert evaluator. Compare the model's answer with the ground truth answer and determine if the model's answer is correct."""

GLOBEM_ACCURACY_CHECK_USER = """Ground Truth Answer: {ground_truth}
Model's Answer: {model_answer}

Respond with only 'YES' if the model's answer is correct, or 'NO' if it is not correct."""


# =============================================================================
# Prompt Factory
# =============================================================================

def get_prompts(scenario: str) -> dict:
    """Get prompts for a specific scenario."""
    if scenario == "mimic":
        return {
            "message_system": MIMIC_CONTEXT_EVAL_SYSTEM,
            "message_user": MIMIC_CONTEXT_EVAL_USER,
            "chat_system": MIMIC_CHAT_SYSTEM,
            "chat_user": MIMIC_CHAT_USER,
            "use_qa_accuracy": False,
        }
    elif scenario == "10k":
        return {
            "message_system": TENK_CONTEXT_EVAL_SYSTEM,
            "message_user": TENK_CONTEXT_EVAL_USER,
            "chat_system": TENK_CHAT_SYSTEM,
            "chat_user": TENK_CHAT_USER,
            "use_qa_accuracy": False,
        }
    elif scenario == "globem":
        return {
            "message_system": GLOBEM_MESSAGE_SYSTEM,
            "message_user": GLOBEM_MESSAGE_USER,
            "chat_system": GLOBEM_CHAT_SYSTEM,
            "chat_user": GLOBEM_CHAT_USER,
            "accuracy_system": GLOBEM_ACCURACY_CHECK_SYSTEM,
            "accuracy_user": GLOBEM_ACCURACY_CHECK_USER,
            "use_qa_accuracy": True,
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def get_entity_prefix(scenario: str) -> str:
    """Get entity prefix for a specific scenario."""
    prefixes = {
        "mimic": "patient",
        "10k": "company",
        "globem": "user"
    }
    return prefixes.get(scenario, "entity")
