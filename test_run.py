import sys

import asyncio
from agents.agents.agents.specialized.image_agent.image_agent import ImageEditingAgent
import os
from datetime import datetime

async def test_image_editing():
    """
    测试ImageEditingAgent的图像编辑功能
    """
    
    # 1. 初始化Agent
    print("🚀 初始化ImageEditingAgent...")
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
        streaming="console",  # 实时显示处理过程
    )
    
    # 2. 准备测试用例
    test_cases = [
        {
            "name": "替换动物",
            "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            "instruction": "Find the animal in the image and replace it with a cute panda"
        },
        {
            "name": "移除物体",
            "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",  # 替换为你的图片URL
            "instruction": "Remove the person from the image and fill the area naturally"
        },
        {
            "name": "更换背景",
            "image_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",  # 替换为你的图片URL
            "instruction": "Change the background to a beautiful beach sunset"
        }
    ]
    
    # 3. 运行测试
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"📝 测试用例 {i+1}: {test_case['name']}")
        print(f"🖼️  图片URL: {test_case['image_url']}")
        print(f"📋 指令: {test_case['instruction']}")
        print(f"{'='*60}\n")
        
        # 构建消息
        messages_list = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": test_case["image_url"]
                            },
                            {
                                "type": "text",
                                "text": test_case["instruction"]
                            }
                        ]
                    }
                ]
            }
        ]
        
        try:
            # 运行Agent
            print("⏳ 处理中...")
            await agent.run_async(
                start_messages=messages_list,
                max_steps=4,  # 最多4步完成任务
                num_chains=1,
                enable_streaming=True
            )
            
            # 打印结果
            print("\n✅ 处理完成！")
            
            # 获取最终的消息
            agent_messages = agent.get_messages()
            if agent_messages and len(agent_messages) > 0:
                last_messages = agent_messages[0]["messages"]
                
                # 查找最终生成的图片ID
                for msg in last_messages:
                    if msg.get("role") == "tool":
                        content = msg.get("content", [])
                        for item in content:
                            if isinstance(item, dict) and "Image Id:" in item.get("text", ""):
                                # 提取图片ID
                                import re
                                match = re.search(r'Image Id:\s*(\d+)', item["text"])
                                if match:
                                    image_id = match.group(1)
                                    
                                    # 保存结果图片
                                    output_dir = "test_outputs"
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_path = f"{output_dir}/test_{i+1}_{timestamp}.png"
                                    
                                    agent.save_image(image_id, output_path)
                                    print(f"💾 结果已保存到: {output_path}")
                                    break
            
            # 打印完整的对话历史
            print("\n📜 对话历史:")
            agent.print_messages(index=0)
            
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 等待用户确认继续
        if i < len(test_cases) - 1:
            input("\n按Enter继续下一个测试...")
    
    print("\n🎉 所有测试完成！")

async def test_specific_function():
    """
    测试特定功能的示例
    """
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
        streaming="console"
    )
    
    # 测试1: 物体检测并替换
    print("\n🔍 测试: 检测并替换物体")
    messages = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Detect the dog in the image and replace it with a cat. Make sure the cat looks natural in the scene."
                        }
                    ]
                }
            ]
        }
    ]
    
    await agent.run_async(
        start_messages=messages,
        max_steps=3,
        num_chains=1,
        enable_streaming=True
    )
    
    # 显示结果
    agent.print_messages(index=0)

async def interactive_test():
    """
    交互式测试 - 允许用户输入自定义的图片URL和指令
    """
    print("\n🎨 ImageEditingAgent 交互式测试")
    print("="*60)
    
    agent = ImageEditingAgent(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        template="qwen2.5-vl-system-tool",
        backend="async_vllm",
        streaming="console"
    )
    
    while True:
        print("\n请输入测试信息（输入 'quit' 退出）:")
        
        image_url = input("图片URL: ").strip()
        if image_url.lower() == 'quit':
            break
            
        instruction = input("编辑指令: ").strip()
        if instruction.lower() == 'quit':
            break
        
        messages_list = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_url
                            },
                            {
                                "type": "text",
                                "text": instruction
                            }
                        ]
                    }
                ]
            }
        ]
        
        try:
            print("\n⏳ 处理中...")
            await agent.run_async(
                start_messages=messages_list,
                max_steps=4,
                num_chains=1,
                enable_streaming=True
            )
            
            print("\n✅ 处理完成！")
            agent.print_messages(index=0)
            
            # 询问是否保存结果
            save = input("\n是否保存结果图片？(y/n): ").strip().lower()
            if save == 'y':
                # 这里添加保存逻辑
                pass
                
        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")
    
    print("\n👋 再见！")

if __name__ == "__main__":
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 运行预定义测试用例")
    print("2. 测试特定功能")
    print("3. 交互式测试")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        asyncio.run(test_image_editing())
    elif choice == "2":
        asyncio.run(test_specific_function())
    elif choice == "3":
        asyncio.run(interactive_test())
    else:
        print("无效选择")