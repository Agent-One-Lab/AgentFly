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