from chat_bricks import Template, register_template

register_template(
    Template(
        name="search-r1",
        system_template="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{system_message} Question: ",
        user_template="{content}\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        tool_template="<information>{observation}</information>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="action-agent",
        system_template="<|im_start|>system\n{system_message}\n<|im_end|>\n",
        user_template="<|im_start|>user\n{content}\n<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        tool_template="<observation>{observation}</observation>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen3-instruct-no-tool",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen3-think-no-tool",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n<think>",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen3-think",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_template_with_tools="""<|im_start|>system\n{system_message}# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools><|im_end|>\n""",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n<think>",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen3-miniswe",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n<think>\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

XLMToolCallTemplate = """<|im_start|>system\n{system_message}

You have access to external functions (tools). When necessary, you may call them to help answer the user's query.

AVAILABLE TOOLS:

{tools}

Tool Call Format (STRICT)

If you decide to call a function, you MUST output **exactly** the following structure:

<tool_call>
<function=FUNCTION_NAME>
<parameter=PARAMETER_NAME_1>
VALUE_1
</parameter>
<parameter=PARAMETER_NAME_2>
VALUE_2
</parameter>
...
</function>
</tool_call>

Rules

1. Use XML format exactly as specified.

2. `<function=...>` must contain the function name. Each argument must be wrapped in its own `<parameter=...>` block

3. Always include all required parameters for the function. Use exact parameter names as defined in the tool schema


Example

<tool_call>
<function=search>
<parameter=query>
weather in San Francisco
</parameter>
</function>
</tool_call>

<|im_end|>\n
"""

register_template(
    Template(
        name="qwen-xml",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_template_with_tools=XLMToolCallTemplate,
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)

register_template(
    Template(
        name="qwen-xml-think",
        system_template="<|im_start|>system\n{system_message}<|im_end|>\n",
        system_template_with_tools=XLMToolCallTemplate,
        user_template="<|im_start|>user\n{content}<|im_end|>\n",
        assistant_template="<|im_start|>assistant\n<think>{content}<|im_end|>\n",
        generation_prompt="<|im_start|>assistant\n<think>",
        tool_template="<|im_start|>user\n<tool_response>\n{observation}\n</tool_response><|im_end|>\n",
        stop_words=["<|im_end|>"],
    )
)