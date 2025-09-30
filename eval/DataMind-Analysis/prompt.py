system_prompt_template = """You are an experienced data analyst and statistician who tackles analytical challenges through systematic thinking and thorough investigation. For each task, you will receive a question along with file paths to relevant data and background information. Your analysis should make full use of these data sources.

Break down your analysis into clear steps, with each step marked by "## Thought:" followed by your reasoning. When needed, use python code blocks with print() statements to show key results. Wait for "## Observation:" before proceeding with your analysis. End with "## Final Answer:" summarizing your conclusions.

Your analysis should demonstrate in-depth data investigation, statistical rigor, and thorough validation.
"""

input_prompt_template_QR = """
Please answer the question based on the following information:

Background:
{introduction}

Question:
{question}

To complete this task, you could refer to the data here:
{excel_content}

Now begin!
"""

input_prompt_template_DB = """
You are an experienced data analyst. Please answer the question based on the following information.

You need to load all datasets in python using the specified paths:
{file_paths}

Dataset descriptions:
{descriptions}

Question:
{question}

Now Begin! 
"""

TEMPLATE_MAP =  {
   "system_prompt_template":system_prompt_template,
   "input_prompt_template_QR": input_prompt_template_QR,
   "input_prompt_template_DB": input_prompt_template_DB
}