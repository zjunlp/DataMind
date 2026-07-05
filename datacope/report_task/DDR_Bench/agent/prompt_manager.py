#!/usr/bin/env python3
"""
Prompt management for the autonomous data analysis agent
"""

class PromptManager:
    """Manages system prompts and task-specific instructions"""
    
    def __init__(self, auto_finish: bool = True, skill_content: str = "", scenario: str = ""):
        self.auto_finish = auto_finish
        self.skill_content = (skill_content or "").strip()
        self.scenario = (scenario or "").lower()
        
        # Base system prompt for autonomous operation
        self.base_system_prompt = """You are an autonomous data analysis agent. Your task is to keep exploring and analyzing data for the given task.

IMPORTANT INSTRUCTIONS:
1. always respond in a ReAct style: return what you're thinking and planning to do, and then call the appropriate tool. RETURN BOTH THE TEXT CONTENT AND THE TOOL CALL.
2. you can only call one tool every turn.
3. your reasoning should contain insights derived from last turn's tool call results. BUT DO NOT INCLUDE ANY INSIGHTS OR REASONING IN THE TOOL CALLS. TOOLS ARE ONLY FOR DATA EXPLORE.
4. you should try your best to use the tools to get more information. keep exploring, build more and more complex params as turns go on and you will discover more in the data.
5. first use the tools to check what data is available to you.
"""
        
        # Task completion instruction (conditional)
        self.finish_instruction = """

TASK COMPLETION:
When you can not gather more information, send a message that starts with "FINISH:" followed by your all insights collected from the whole dialogue and tool calls. 
Only use "FINISH:" when you are absolutely certain that no more information can be gathered. 
Carefully use "FINISH:" in your message since it will immediately end the session. Think twice before using it."""

        self.qa_pair_direction_guidance = {
            "10k": """The conclusion must be supported by multiple pieces of evidence gathered through tools. The question should be broad enough to require synthesis, specific enough to be judged against evidence, and aligned with the answer. Keep the answer concise, usually one sentence and at most two short sentences. Prefer qualitative synthesis over dense numeric recap. A single evidence item may support more than one conclusion, but each submitted QA pair should capture a distinct angle.

            Possible QA directions include, but are not limited to, broad trends, causes or constraints, exposures and responses, important dependencies, operating or cost pressures, disclosure or accounting judgements, comparability issues and risks affecting flexibility or performance. Treat this as loose inspiration rather than set categories. Avoid asking the same kind of question repeatedly; vary both the perspective and wording naturally based on what the exploration actually reveals.""",

            "mimic": """The conclusion must be grounded in one or more pieces of evidence gathered through tools. A single clear record may be enough for a focused conclusion, while related records, joins, or small aggregations may support broader conclusions. Keep the answer concise, usually as a short value, compact list, or brief phrase; add explanation only when the evidence calls for it.

            Possible QA directions include, but are not limited to, patient profile, care episodes, clinical services, clinical findings, documented actions, and measurements, with perspectives including documentation, progression, timing, comparison, variation, and outcomes. Treat this as loose inspiration rather than set categories. Avoid asking the same kind of question repeatedly; vary both the perspective and wording naturally based on what the exploration actually reveals.""",

            "globem": """The conclusion must be grounded in one or more pieces of evidence gathered through tools. Some conclusions may come from a direct comparison, while others may require summarizing patterns across records or time windows. Keep the answer concise, usually as a short directional or comparative phrase.

            Possible QA directions include, but are not limited to, psychological state, social interaction, coping style, daily activity, and passive sensing signals; consider how these change over time or between observations. Treat this as loose inspiration rather than set categories. Avoid asking the same kind of question repeatedly; vary metrics and dimensions based on what the exploration actually reveals.""",
        }
        self.qa_pair_style_examples = {
            "10k": 
            """q: How do the recent usage trends and capacity constraints affect the reliability of the service?
            a: The service appears to be handling higher usage more consistently, but recurring capacity limits and maintenance delays still make reliability sensitive to demand spikes or resource shortages.""",

            "mimic": 
            """q: Which care team was primarily associated with the documented encounter?
            a: The record points to a medical service, with another specialty appearing later in the care timeline.""",
            
            "globem": """
            q: How does the entity's measured state change between the two observations?
            a: It shows a modest improvement.""",
        }
        self.qa_pair_system_prompt = """You are an autonomous data research agent. Your task is to keep exploring the data and generate high-quality QA pairs when the evidence supports a meaningful conclusion.

Your generated QA pair must contain:
- q: a question that would naturally elicit the conclusion.
- a: the key conclusion, written as a concise answer.

{qa_pair_direction_guidance}

STYLE EXAMPLE:
{qa_pair_style_example}
Follow this example's form style, but make each submitted QA pair more specific and informative based on the evidence you actually gathered.

IMPORTANT INSTRUCTIONS:
1. always respond in a ReAct style: return what you're thinking and planning to do, and then call the appropriate tool. RETURN BOTH THE TEXT CONTENT AND THE TOOL CALL.
2. you can only call one tool every turn.
3. your reasoning should contain evidence derived from last turn's tool call results. BUT DO NOT INCLUDE ANY EVIDENCE OR REASONING IN THE TOOL CALLS. Data-exploration tools are only for gathering evidence; submit_qa_pair is only for emitting the q and a fields.
4. you should try your best to use the tools to get more information. keep exploring, build more and more complex params as turns go on and you will discover more in the data.
5. when the gathered evidence is strong enough for a non-trivial conclusion, call "submit_qa_pair" tool for exactly one QA pair. Do not stop after only a few pairs; keep exploring and submit a reasonably comprehensive set of distinct, well-supported QA pairs before finishing.
6. first use the tools to check what data is available to you.
"""


        self.qa_pair_finish_instruction = """

TASK COMPLETION:
When you can not gather more information or can not generate additional distinct, well-supported QA pairs, send a message that starts with "FINISH:" followed by a concise inventory of the QA pairs you generated and any remaining high-level coverage notes.
Only use "FINISH:" when you are absolutely certain that no more meaningful QA pairs can be generated from the available data.
Carefully use "FINISH:" in your message since it will immediately end the session. Think twice before using it."""
    
    def build_system_prompt_with_task(self, task: str) -> str:
        """Build system prompt with specific task"""
        enhanced_prompt = self.base_system_prompt
        
        # Add finish instruction only if auto_finish is enabled
        if self.auto_finish:
            enhanced_prompt += self.finish_instruction
        
        enhanced_prompt += f"\n\nYOUR TASK: {task}"
        if self.skill_content:
            enhanced_prompt += (
                "\n\nThe following user-selected skills are additional task guidance. "
                "Use them when relevant, but still strictly follow all core system "
                "instructions, tool constraints, and output format requirements."
            )
            enhanced_prompt += "\n\nSKILL CONTEXT:\n"
            enhanced_prompt += self.skill_content
        enhanced_prompt += "\n\nAnalyze the task and use the available tools to accomplish it step by step."
        enhanced_prompt += "\n\nALWAYS RETURN in the format of text content with a tool call, no just return the tool call."
        
        return enhanced_prompt

    def build_qa_pair_system_prompt_with_task(self, task: str) -> str:
        """Build QA-pair generation system prompt with specific task."""
        direction_guidance = self.qa_pair_direction_guidance.get(self.scenario, "")
        style_example = self.qa_pair_style_examples.get(self.scenario, "")
        enhanced_prompt = self.qa_pair_system_prompt.format(
            qa_pair_direction_guidance=direction_guidance,
            qa_pair_style_example=style_example,
        )

        # Add finish instruction only if auto_finish is enabled
        if self.auto_finish:
            enhanced_prompt += self.qa_pair_finish_instruction

        enhanced_prompt += f"\n\nYOUR TASK: {task}"
        if self.skill_content:
            enhanced_prompt += (
                "\n\nThe following user-selected skills are additional task guidance. "
                "Use them when relevant, but still strictly follow all core system "
                "instructions, tool constraints, and output format requirements."
            )
            enhanced_prompt += "\n\nSKILL CONTEXT:\n"
            enhanced_prompt += self.skill_content
        enhanced_prompt += "\n\nAnalyze the task and use the available tools to accomplish it step by step."
        enhanced_prompt += "\n\nALWAYS RETURN in the format of text content with a tool call, no just return the tool call."

        return enhanced_prompt
    
    def build_insight_prompt(self, assistant_content: str, user_content: str, task: str) -> str:
        """Build prompt for generating insights from tool execution"""
        return f"""Based on the following tool execution, provide a brief insight about what was discovered or learned:

The reason and action to use the tool:
{assistant_content}

Tool execution result:
{user_content}

Provide a concise insight (1-3 sentences) about what this reveals:
1. It has to be related to the task: {task}.
2. If there is no insight or error in the tool execution, respond with 'NO INSIGHT'.
3. If it only use the data description tools (e.g. tools like list_files, describe_table, get_database_info, get_field_description), respond with 'NO INSIGHT'.
4. The insight from data should answer the question raised in the reason to execute this tool. Focus on this point.
5. Keep all the data or statitics needed in your generated insight.
ONLY respond with the insight."""
    
    def get_insight_system_prompt(self) -> str:
        """Get system prompt for insight generation"""
        return "You are an expert data analyst. Provide concise, actionable insights based on tool execution results."
    
    def build_final_summary_prompt(self, messages: list) -> str:
        """Build prompt for generating final summary from chat message list"""
        # Format conversation history from chat message list
        conversation_text = ""
        for msg in messages:
            if msg.get("role") != "system":
                conversation_text += f"{msg.get('role', '').upper()}: {msg.get('content', '')}\n"
                if msg.get('tool_call'):
                    conversation_text += f"TOOL_CALL: {msg['tool_call']}\n"
                if msg.get('tool_result'):
                    conversation_text += f"TOOL_RESULT: {msg['tool_result']}\n"

        skill_text = ""
        if self.skill_content:
            skill_text = (
                "\nThe following user-selected skills are additional task guidance. "
                "Use them when relevant, but still strictly follow all core system "
                "instructions, tool constraints, and output format requirements."
                f"\n\nSKILL CONTEXT (VERBATIM):\n{self.skill_content}\n"
            )

        return f"""Based on the entire conversation history below, provide a comprehensive final summary of your analysis and findings.
{skill_text}

CONVERSATION HISTORY:
{conversation_text}

Please provide a detailed final summary that includes all insights collected from the whole dialogue and tool calls. The summary should be no more than 8192 tokens. Format your response as: "FINISH: [your comprehensive summary here]"
"""

    def build_qa_pair_final_summary_prompt(self, messages: list) -> str:
        """Build final summary prompt for QA-pair generation sessions."""
        conversation_text = ""
        for msg in messages:
            if msg.get("role") != "system":
                conversation_text += f"{msg.get('role', '').upper()}: {msg.get('content', '')}\n"
                if msg.get('tool_call'):
                    conversation_text += f"TOOL_CALL: {msg['tool_call']}\n"
                if msg.get('tool_result'):
                    conversation_text += f"TOOL_RESULT: {msg['tool_result']}\n"

        skill_text = ""
        if self.skill_content:
            skill_text = (
                "\nThe following user-selected skills are additional task guidance. "
                "Use them when relevant, but still strictly follow all core system "
                "instructions, tool constraints, and output format requirements."
                f"\n\nSKILL CONTEXT (VERBATIM):\n{self.skill_content}\n"
            )

        return f"""Based on the entire conversation history below, provide a final summary for a QA-pair generation session.
{skill_text}

CONVERSATION HISTORY:
{conversation_text}

Please summarize:
- QA pairs that were generated or submitted, preserving their q/a content if present.
- The main evidence themes used to support those QA pairs.
- Any remaining coverage gaps or promising areas that were explored but did not produce a distinct, well-supported QA pair.

Do not invent new QA pairs in this final summary. Do not convert unsupported notes into q/a form. The summary should be no more than 8192 tokens. Format your response as: "FINISH: [your checklist generation summary here]"
"""
    
    def get_final_summary_system_prompt(self) -> str:
        """Get system prompt for final summary generation"""
        return "You are an expert data analyst. Provide comprehensive final summaries based on complete conversation histories."

    def get_qa_pair_final_summary_system_prompt(self) -> str:
        """Get system prompt for QA-pair final summary generation."""
        return "You are an expert QA curation analyst. Summarize generated QA pairs and their evidence coverage without inventing new pairs."
