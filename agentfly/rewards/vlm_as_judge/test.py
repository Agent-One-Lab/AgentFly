#!/usr/bin/env python
"""Fixed test script for VLM with proper async handling"""

import asyncio
import json
import re
from openai import AsyncOpenAI

# Your complete question data
QUESTION_DATA = {
    "question": "A 6 kg cube of polished marble, with a side length of 0.3 meters, is released from the top of a 40-degree inclined plane inside a 4-meter-long, 3-meter-wide, and 2-meter-high metal chamber. The plane is lined with a layer of felt, providing a friction coefficient of 0.3. The chamber's interior is heated to 30°C, causing a gentle convection current that introduces a slight, variable force acting against the marble's descent. Over 16 seconds, the marble cube slides down the 2.5-meter plane, the friction and convection currents adding complexity to its motion.",
    "Level": 3,
    "vlm_questions": {
        "enableAnnotator": "Yes",
        "summarize": "A 6 kg cube of polished marble is released from the top of a 40-degree inclined plane inside a metal chamber. The plane is lined with felt, providing friction, while convection currents from the heated chamber oppose the cube's descent. Over 16 seconds, the cube slides down the 2.5-meter plane, with friction and convection adding complexity to its motion.",
        "vlm_questions": [
            {
                "index": "1",
                "question": "A cube is released from the top of an inclined plane.",
                "weight": 1.0
            },
            {
                "index": "2",
                "question": "The inclined plane is inside a chamber.",
                "weight": 0.8
            },
            {
                "index": "3",
                "question": "The cube slides down the inclined plane over a period.",
                "weight": 0.7
            },
            {
                "index": "4",
                "question": "The cube's motion is influenced by forces acting against its descent.",
                "weight": 0.9
            },
            {
                "index": "5",
                "question": "The cube descends the entire length of the inclined plane.",
                "weight": 0.6
            }
        ]
    }
}

def extract_results_from_json_response(response):
    """Extract results and calculate simple reward from the JSON response"""
    try:
        # Try to find and parse JSON array
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result_list = json.loads(json_str)
            
            if result_list and len(result_list) > 0:
                scores = []
                confidence_scores = []
                results = []
                
                for result in result_list:
                    # Get result and confidence
                    result_value = result.get("result", "Not sure")
                    confidence = int(result.get("confidence_score", "1"))
                    
                    # Simple scoring logic
                    if result_value == "True":
                        score = 1.0
                    elif result_value == "False":
                        score = 0.0
                    else:  # "Not sure"
                        if confidence >= 4:
                            score = 0.0  # High confidence "Not sure" -> False
                        else:
                            score = 1.0  # Low confidence "Not sure" -> True
                    
                    scores.append(score)
                    confidence_scores.append(confidence)
                    results.append(result_value)
                
                # Calculate mean of all scores
                reward = sum(scores) / len(scores) if scores else 0.5
                return reward, confidence_scores, results, result_list
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    
    return 0.5, [], [], []

def create_vlm_judge_prompt(questions_data, summarize=None):
    """Create VLM judge prompt from question data"""
    
    all_questions = ""
    
    # Extract questions from the data structure
    if isinstance(questions_data, dict) and "vlm_questions" in questions_data:
        vlm_data = questions_data["vlm_questions"]
        if isinstance(vlm_data, dict) and "vlm_questions" in vlm_data:
            questions_list = vlm_data["vlm_questions"]
            for q in questions_list:
                idx = q.get("index", "")
                question = q.get("question", "")
                all_questions += f"{idx}. {question}\n"
            if summarize is None:
                summarize = vlm_data.get("summarize", "")
    
    all_questions = all_questions.strip()
    
    if summarize is None:
        summarize = "Evaluate the visual content based on the questions provided."
    
    # VLM Judge Prompt Template (shortened for clarity)
    prompt = f"""You are given a set of visual verification questions and a description of objects and motion observed in an input medium (e.g., image or video).

Your task is to **evaluate each question** based on whether it is **correctly reflected in the visual content**.

**Input**:
- Questions: 
{all_questions}
- Object and motion description: {summarize}

**For each question**, return:
- `"index"`: the question index
- `"question"`: the full question text
- `"analysis"`: your reasoning process and visual inference
- `"result"`: one of `"True"`, `"False"`, or `"Not sure"`
- `"confidence_score"`: an integer from 1 (very uncertain) to 5 (very certain)

**Output Format**:
Return a JSON list like this:
[
    {{
        "index": "1",
        "question": "A cube is released from the top of an inclined plane.",
        "analysis": "I can see a cube at the top of an inclined surface.",
        "result": "True",
        "confidence_score": "5"
    }},
    ...
]"""
    
    return prompt

async def test_vlm_async():
    """Async function to test VLM"""
    
    # Direct connection to your server
    client = AsyncOpenAI(
        base_url="http://10.24.1.23:8000/v1",
        api_key="token-abc123"
    )
    
    # Video path - UPDATE THIS WITH YOUR VIDEO
    video_path = "/mnt/sharefs/users/xuezhi.liang/save_video/batch_0/video_462/video_462_3.mp4"
    
    print("=" * 70)
    print("VLM Test with Async Function")
    print("=" * 70)
    print(f"Server: http://10.24.1.23:8000")
    print(f"Video: {video_path}")
    print(f"Number of questions: {len(QUESTION_DATA['vlm_questions']['vlm_questions'])}")
    print("=" * 70)
    
    # Create the prompt
    prompt = create_vlm_judge_prompt(QUESTION_DATA)
    
    # Add video reference
    prompt_with_video = f"<video>{video_path}</video>\n\n{prompt}"
    
    try:
        print("\nSending request to VLM server...")
        
        messages = [
            {
                "role": "user",
                "content": prompt_with_video
            }
        ]
        
        # Make the API call - THIS MUST BE IN AN ASYNC FUNCTION
        response = await client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Get the response
        output_text = response.choices[0].message.content
        print("\n" + "=" * 70)
        print("Raw VLM Response:")
        print("-" * 70)
        print(output_text[:1000])  # First 1000 chars
        if len(output_text) > 1000:
            print("... (truncated)")
        print("-" * 70)
        
        # Parse and calculate scores
        reward, confidence_scores, results, full_results = extract_results_from_json_response(output_text)
        
        if results:
            print("\n" + "=" * 70)
            print("Parsed Results and Scoring:")
            print("=" * 70)
            
            for i, (res, conf) in enumerate(zip(results, confidence_scores)):
                if res == "True":
                    score = 1.0
                elif res == "False":
                    score = 0.0
                else:  # "Not sure"
                    score = 0.0 if conf >= 4 else 1.0
                
                print(f"\nQ{i+1}: {QUESTION_DATA['vlm_questions']['vlm_questions'][i]['question']}")
                print(f"  Result: {res}")
                print(f"  Confidence: {conf}/5")
                print(f"  → Score: {score}")
            
            print("\n" + "=" * 70)
            print("Final Scoring:")
            print("=" * 70)
            print(f"Mean Reward: {reward:.2f}")
            print("\nScoring Logic Applied:")
            print("  TRUE → 1.0")
            print("  FALSE → 0.0")
            print("  'Not sure' with confidence >= 4 → 0.0")
            print("  'Not sure' with confidence < 4 → 1.0")
            
            return {
                "reward": reward,
                "results": results,
                "confidence_scores": confidence_scores
            }
        else:
            print("\nNo valid results parsed from response")
            return None
        
    except Exception as e:
        print(f"\nError calling VLM server: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the async test"""
    print("Starting VLM Test")
    print("This test uses the new scoring logic without weights\n")
    
    # Run the async function
    result = asyncio.run(test_vlm_async())
    
    if result:
        print("\n" + "=" * 70)
        print("Test Completed Successfully!")
        print(f"Final Reward: {result['reward']:.2f}")
    else:
        print("\n" + "=" * 70)
        print("Test completed with errors. Check the output above.")

if __name__ == "__main__":
    main()