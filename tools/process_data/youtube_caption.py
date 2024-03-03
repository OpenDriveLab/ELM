import os
import random

def call_prompt(idx):
    prompt = []
    prompt.append(f"Determine the spatial coordinates in 3D corresponding to the 2D pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What is the world-space position of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}> in 3D?")
    prompt.append(f"Provide the 3D location of the 2D point <c, CAM_FRONT, {coor[0]}, {coor[1]}> in the image.")
    prompt.append(f"Calculate the 3D spatial coordinates of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"Calculate the 3D world location of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What are the 3D world-space coordinates of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>?")

    return prompt[idx]

def call_prompt1(idx):
    prompt = []
    prompt.append(f"Determine the spatial coordinates in 3D corresponding to the 2D pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What is the world-space position of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}> in 3D?")
    prompt.append(f"Provide the 3D location of the 2D point <c, CAM_FRONT, {coor[0]}, {coor[1]}> in the image.")
    prompt.append(f"Calculate the 3D spatial coordinates of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"Calculate the 3D world location of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What are the 3D world-space coordinates of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>?")

    return prompt[idx]

def call_prompt2(idx):
    prompt = []
    prompt.append(f"Determine the spatial coordinates in 3D corresponding to the 2D pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What is the world-space position of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}> in 3D?")
    prompt.append(f"Provide the 3D location of the 2D point <c, CAM_FRONT, {coor[0]}, {coor[1]}> in the image.")
    prompt.append(f"Calculate the 3D spatial coordinates of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"Calculate the 3D world location of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What are the 3D world-space coordinates of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>?")

    return prompt[idx]

def call_prompt3(idx):
    prompt = []
    prompt.append(f"Determine the spatial coordinates in 3D corresponding to the 2D pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What is the world-space position of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}> in 3D?")
    prompt.append(f"Provide the 3D location of the 2D point <c, CAM_FRONT, {coor[0]}, {coor[1]}> in the image.")
    prompt.append(f"Calculate the 3D spatial coordinates of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"Calculate the 3D world location of the pixel with 2D coordinates <c, CAM_FRONT, {coor[0]}, {coor[1]}>.")
    prompt.append(f"What are the 3D world-space coordinates of the pixel at <c, CAM_FRONT, {coor[0]}, {coor[1]}>?")

    return prompt[idx]


if __name__ == "__main__":
    root = "/cpfs01/user/huanglinyan/projects/LLaMA-Adapter/llama_adapter_v2_multimodal7b"
    data = []
    for i in range(8):
        with open(os.path.join(root, f"waymo-{data_split}.json"), "r") as f:
            data.extend(json.load(f))

    questions = []
    answers = []
    image_paths = []
    for info in data:
        answer = info["caption"]
        answer1 = answer.split("2.")[0].split("1.")[-1]
        answer2 = answer.split("3.")[0].split("2.")[-1]
        answer3 = answer.split("2.")[-1]

        idx1 = random.randint(0, 10000)
        idx2 = random.randint(0, 10000)
        idx3 = random.randint(0, 10000)
        question1 = call_prompt1(idx1)
        question2 = call_prompt2(idx2)
        question3 = call_prompt3(idx3)

        image_path = info["img_path"]

        # output the results
        questions.append(question1)
        questions.append(question2)
        questions.append(question3)

        answers.append(answer1)
        answers.append(answer2)
        answers.append(answer3)

        image_paths.extend([image_path]*3)
    
    dump_files = {}
    dump_files["answers"] = answers
    dump_files["image_paths"] = image_paths
    dump_files["questions"] = questions
    with open("waymo_data.json", "wb") as f:
        json.dump(dump_files, f, indent=4)  

