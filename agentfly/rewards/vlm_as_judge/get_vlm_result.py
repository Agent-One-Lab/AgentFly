from utils import LLMClient
import asyncio  
import time
import json
import os
import re
from tqdm import tqdm
import shutil
from llm_utils import clean_response, extract_triple_single_quote_json
import sys
savejsonl = True
json_arr = []
# not good
# First identity each object described in {summarize} and their corresponds in the video, and then answer the question based on the visual content.

# Simple usage
def create_prompt(summarize, all_questions):
    return f"""You are given a set of visual verification questions and a description of objects and motion observed in an input medium (e.g., image or video).

Your task is to **evaluate each question** based on whether it is **correctly reflected in the visual content**, considering visual cues, shape changes from viewpoint, and possible symbolic representations.


---

 **Visual Reasoning Guidelines**:

1. **Perspective Awareness**:  
   Objects may appear different based on viewpoint. For example:
   - A **cylinder** may look like a **circle (top view)** or a **rectangle/square (side view)**.
   - A **circular path** may appear as a **wave-like curve or straight line** in 2D projection.

2. **Symbolic Representations**:  
   Common simplifications may be used. You should **reasonably infer** their meaning:
   - A series of **dots or circles** may represent **foam markers** or control points.
   - A **rectangle** may represent a **container** (e.g., cylindrical viewed from the side).
   - A **line** may represent a **rubber mat** or constraint boundary.
   - The object and track specifics might do not match directly, if the motion can be interpreted correctly, it is still true.
   - It might use color to represent different objects, such as a green line to represent the flat surface is covered with a felt-like material.
   - The rotation of the object might cannot be judged from the video, but the motion can be interpreted correctly, it is still true.

3. **Container Boundaries**:
   - If **no container is drawn**, you may assume the **video frame itself is the container boundary**.
   - If a **container is visible**, treat it as **transparent** if inner content is visible.
   - If the object is not visible, you should not assume it is in the container.

4. **Focus on Shape & Position**, **not material**:
   - Ignore assumptions about object **material**, **color**, or **texture**.
   - Base your decisions entirely on **observable geometry** (e.g., shape, layout, structure) and **motion** (e.g., direction, trajectory).
   - Use visible movement and positioning to judge truthfulness — even if the object type is unknown.
   - If the described motion is **sliding down a slope**, but the video shows an **upward movement**, the result should be `"False"` — regardless of material or appearance.
   - Make geometric and motion-based reasoning the core of your judgment, even when objects are **partially occluded**.

5. **Occlusion Handling**:
   - If an object is **partially blocked**, assess based on surrounding evidence whether its state or motion can still be inferred.

6. **Avoid excessive uncertainty**:
   - If there is enough visual context and logical structure, make a **confident judgment**.
   - Use "Not sure" only when the evidence is **truly insufficient or ambiguous**.

---

 **Input**:
- Questions: {all_questions}
- Object and motion description: {summarize}

---

 **For each question**, return:
- `"index"`: the question index
- `"question"`: the full question text
- `"analysis"`: your reasoning process and visual inference
- `"result"`: one of `"True"`, `"False"`, or `"Not sure"`
- `"confidence_score"`: an integer from 1 (very uncertain) to 5 (very certain)

---

**Output Format**:
Return a JSON list like this:
[
    {{
        "index": "1",
        "question": "The ball rolls along the circular path.",
        "analysis": "The object follows a closed curve consistent with a circular path from the top view.",
        "result": "True",
        "confidence_score": "5"
    }},
    ...
]
"""

def resize_to_constraints(W, H, max_area=280*480, divisor=28):
    aspect_ratio = W / H
    candidates = []

    # 遍历满足条件的所有可能的尺寸
    for target_h in range(divisor, 2048, divisor):
        target_w = int(target_h * aspect_ratio)
        if target_w <= 0:
            continue
        area = target_w * target_h
        if area <= max_area:
            candidates.append((area, target_w, target_h))

    if not candidates:
        raise ValueError("No valid size found under constraints.")

    # 选面积最大的那一组
    _, final_w, final_h = max(candidates, key=lambda x: x[0])
    return final_w, final_h

async def main():
    valid_question_num = 0
    Model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    index = 0
    client = LLMClient(model=Model_name, timeout_seconds=10000, temperature=0.0)
    for i in range(0, 24481, 1):
        folder_id = i // 2500
        json_path = f'result/verify/batch_{folder_id}/video_{i}/video_{i}.json'
        if os.path.exists(f'result/vlm/batch_{folder_id}/video_{i}/'):
            continue
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                item = json.load(f)
            origin_question = item['question']
            
            with open(f'result/video_info{folder_id}.json', 'r', encoding='utf-8') as f:
                video_info = json.load(f)
            try:
                after_verify = item['vlm_result']
            except:
                print(type(item['vlm_result']))
                print(f'after_verify error: verify/batch_{folder_id}/video_{i}/video_{i}.json')
                break
            if after_verify['enableAnnotator'] == 'No' or len(after_verify['vlm_questions']) == 0:
                continue
            valid_question_num += 1
            all_questions = ''
            all_weight= []
            try:
                for vlmindex in range(len(after_verify['vlm_questions'])):
                    subquestion = after_verify['vlm_questions'][vlmindex]['question']
                    all_questions += f'{vlmindex+1}. {subquestion}'
                    all_weight.append(after_verify['vlm_questions'][vlmindex]['weight'])
            except:
                print(json_path)

            video_folder = f'/mnt/sharefs/users/xuezhi.liang/save_video/batch_{folder_id}/video_{i}'
            for video_name in os.listdir(video_folder):
                sub_info = video_info[os.path.splitext(video_name)[0]]
                new_width, new_height = resize_to_constraints(sub_info['width'], sub_info['height'])
                messages = [
                    {
                        "role": "user",
                        "content": 
                            [{
                                "type": "text", 
                                "text": create_prompt(after_verify['summarize'], all_questions)
                            },
                            {
                                "type": "video",
                                "video":f"{video_folder}/{video_name}",
                                "resized_height": new_height,
                                "resized_width": new_width,
                                "fps": 5,
                            }]
                    }]
                    #                             
                if savejsonl:
                    json_arr.append({
                        "messages": messages,
                        "id": i,
                        "document": json_path,
                        "video": f"{video_folder}/{video_name}"
                    })
                else:
                    responses = await client.process_all_inputs([messages], model=Model_name)
                    try:
                        final_json = extract_triple_single_quote_json(responses[0][0])
                        if final_json:
                            print(final_json)
                    except:
                        print(video_name, responses)
        except:
            print(f'error: {json_path}')
            break
    if savejsonl:
        with open(f"run_vlm_fps5_finalbatch_test.jsonl", "w", encoding="utf-8") as f:
            for item in json_arr:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f'valid_question_num: {valid_question_num}')
    return
def transfer_signal(text):
    if '\n' in text:
        text = text[3:-3]
    text = text.replace("\"", "\\\"")
    return text
def process_jsonl():
    total_num = 0
    process_num = 0
    null_num = 0
    batch_index = 0
    with open(f"/mnt/sharefs/users/haonan.li/VLM-as-judge-proj/vllm_server/run_vlm_fps5_finalbatch_responses.jsonl", "r", encoding="utf-8") as f:
        for i, row in tqdm(enumerate(f)):
            item = json.loads(row)
            total_num += 1
            folder_id = int(item['id']) // 2500
            id = item['id']
            document = item['document']
            video = item['video']
            with open(document, "r", encoding="utf-8") as f:
                setting = json.load(f)
            origin_question = setting['question']
            level = setting['Level']
            vlm_questions = setting['vlm_result']['vlm_questions']
            
            weight_arr = []
            for vlmindex in range(len(vlm_questions)):
                weight_arr.append(vlm_questions[vlmindex]['weight'])

            for index, response in enumerate(item['responses']):
                if response is None:
                    null_num += 1
                    continue
                save_name = f"{os.path.basename(video).replace('.mp4', '')}_{batch_index}.json"
                try:
                    result = clean_response(response)
                except:
                    print('first:', response)
                    raise Exception(f'{json_folder}/{save_name}')
                    break
                try:
                    json.loads(result)
                except:
                    filter_result = extract_triple_single_quote_json(result)
                    if len(filter_result) > 0:
                        result = filter_result
                    else:
                        if type(result) != list:
                            result = [result]
                try:  
                    # 处理返回结果中转义字符可能出现的错误
                    result[0] = result[0].replace('\\(', '\\\\(').replace('\\)', '\\\\)')
                    result[0] = result[0].replace('"\n', '",\n')
                    result[0] = re.sub(r',\n\s*}', r'\n}', result[0])
                    result[0] = re.sub(r':(\s)\n', r':\1', result[0])
                    result = json.loads(result[0])
                except:
                    try:
                        matches = re.findall(r': ".*"}?,\s*\n', result[0])
                        confidence_matches = re.findall(r'"confidence_score":\s*"?(\d+)"?,?\n', result[0])
                        new_result = []
                        for jindex in range(0, len(matches), 4):
                            new_result.append({
                                "index": transfer_signal(matches[jindex]),
                                "question": transfer_signal(matches[jindex+1]),
                                "analysis": transfer_signal(matches[jindex+2]),
                                "result": transfer_signal(matches[jindex+3]),
                                "confidence_score": confidence_matches[jindex//4]
                            })
                        result = new_result
                    except:
                        print(f'second error:{json_folder}/{save_name}')
                        # raise Exception(f'second error:{json_folder}/{save_name}')
                json_folder = f'result/vlm/batch_{folder_id}/video_{id}'
                os.makedirs(json_folder, exist_ok=True)
                save_json = {
                    "document": document,
                    "video": video,
                    "weight": weight_arr, 
                    "vlm_result": result,
                }
                with open(f'{json_folder}/{save_name}', 'w', encoding='utf-8') as f:
                    json.dump(save_json, f, indent = 4, ensure_ascii=False)
                process_num += 1
        print(f'process_num: {process_num}, total_num: {total_num}, null_num: {null_num}, error_num: {total_num - process_num - null_num}')

            

if __name__ == "__main__":
    if sys.argv[1] == 'main':
        print('get run.jsonl')
        asyncio.run(main())
    elif sys.argv[1] == 'process_jsonl':
        print('process_jsonl')
