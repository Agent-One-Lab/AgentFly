from agents.agents.agents.templates.utils import tokenize_conversations
from agents.agents.agents.templates.templates import Chat
from transformers import AutoTokenizer
import torch

def contiguous_runs(mask_row: torch.Tensor):
    idxs = (mask_row == 1).nonzero(as_tuple=False).flatten().tolist()
    if not idxs: return []
    runs, s, p = [], idxs[0], idxs[0]
    for k in idxs[1:]:
        if k == p + 1: p = k
        else: runs.append((s, p)); s = p = k
    runs.append((s, p))
    return runs

model = "Qwen/Qwen2.5-3B-Instruct"
template = "qwen2.5-think"  # use your template name
tokenizer = AutoTokenizer.from_pretrained(model)

messages = [
  {"role":"user","content":[{"type":"text","text":"Q"}]},
  {"role":"assistant","content":[{"type":"text","text":"r1"}],"loss":False},
  {"role":"tool","content":[{"type":"text","text":"o1"}]},
  {"role":"assistant","content":[{"type":"text","text":"r2"}],"loss":False},
  {"role":"tool","content":[{"type":"text","text":"o2"}]},
  {"role":"assistant","content":[{"type":"text","text":"r3"}],"loss":True},
]

# Sanity: did the template keep the loss flags?
chat = Chat(template=template, messages=messages, tokenizer=tokenizer)
print("assistant loss flags seen by template:",
      [m.get("loss") for m in chat.messages if m["role"]=="assistant"])

inputs = tokenize_conversations(
  [messages],
  tokenizer=tokenizer,
  conv_template=template,
  max_length=2048,
  return_reward_mask=True,
)

ids = inputs["input_ids"][0]
am  = inputs["action_mask"][0]
rm  = inputs["reward_mask"][0]

print("total_tokens:", ids.size(0))
print("action_mask_sum:", int(am.sum()))
print("reward_mask_sum:", int(rm.sum()))

runs = contiguous_runs(am)
print("num_assistant_spans:", len(runs))
for i, (s, e) in enumerate(runs, 1):
    print(f"span_{i}: [{s},{e}] len={e-s+1} text={tokenizer.decode(ids[s:e+1])!r}")

# This is the exact last assistant response we optimize (should be the only span)
if runs:
    s, e = runs[-1]
    last_txt = tokenizer.decode(ids[s:e+1])
    print("LAST_SPAN (optimized):", (s, e), "len=", e-s+1, "text=", repr(last_txt))