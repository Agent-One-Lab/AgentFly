# 🚀 Predefined Training Examples

Tasks AgentFly has been trained and evaluated on, with links to training reports. The shipped training scripts live under `examples/train_scripts/` — copy the closest match and adapt; see [Build Your Own Task](../start/build_your_own_task.md).

## Tasks

| Task | Model | Report | Status |
|------|-------|--------|--------|
| SearchR1 | Qwen2.5 | [report](https://wandb.ai/AgentRL/Open/reports/SearchR1--VmlldzoxNjYzNzQ0Ng) | ✅ |
| WebShop | Qwen2.5 | [report](https://api.wandb.ai/links/AgentRL/kpmsvggh) | ✅ |
| ScienceWorld | Qwen3-4B-Instruct | [report](https://api.wandb.ai/links/AgentRL/f99omj98) | ✅ |
| SWE | Qwen3-32B | [report](https://wandb.ai/AgentRL/Resource/reports/SWE-OS---VmlldzoxNjUzNjk0Mw?accessToken=x4co1e22ddhkm1qjo791a9blmvt4uqz9jmgytybkq4xtgfwt0u8jjx28wpqcsqex) | ✅ (on going) |
| SimuScene | SFT DeepSeek-R1-Distill-Qwen | [report](https://wandb.ai/AgentRL/SimuScene/reports/SimuScene--VmlldzoxNjYzNzYzMg?accessToken=qe00f9dy59hiu2uyndyn22xl141s37sybnxlp5e5ybryqqahao5mvyra0sbmlf9v) | ✅ |

## Training Curves

Training curves and metrics are logged to WandB for each experiment.

<p align="center">
  <a href="https://wandb.ai/AgentRL/Open">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" width="40" alt="W&B Logo"/><br/>
    <b> 🦋 AgentFly </b><br/>
    <i> 📊 Curves & Logs</i><br/>
  </a>
</p>


<a href="https://wandb.ai/AgentRL/Open" target="_blank"
   style="
     display: inline-flex;
     align-items: center;
     padding: 8px 14px;
     border-radius: 6px;
     background-color: #f6f8fa;
     text-decoration: none;
     font-family: Arial, sans-serif;
     font-size: 14px;
     font-weight: 500;
     color: #333;
     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
     transition: background 0.2s, transform 0.2s;
   "
   onmouseover="this.style.background='#e9ecef'; this.style.transform='translateY(-1px)';"
   onmouseout="this.style.background='#f6f8fa'; this.style.transform='none';">
  <img src="https://wandb.ai/logo.svg" alt="WandB Logo"
       style="height: 20px; margin-right: 8px;" />
  View Training Curves on WandB
</a>
