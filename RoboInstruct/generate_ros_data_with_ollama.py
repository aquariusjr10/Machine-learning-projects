import requests
import json
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"  # Make sure this model is available in Ollama
NUM_TASKS = 500  # How many prompts to generate
OUTPUT_FILE = "Dataset/ros_dataset1.jsonl"



# Function to call Ollama
def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"].strip()


# Step 1: Ask Ollama to generate N robotics instructions
def generate_instructions(n):
    prompt = f"""Generate {n} different high-level robotics instructions for a ROS-based robot system. 
Each instruction should be a natural language command that a user might give, like "Start the lidar node" or "Move forward".
Return them as a numbered list."""
    response = call_ollama(prompt)

    # Extract just the lines with commands
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    instructions = []
    for line in lines:
        if '.' in line:
            try:
                instr = line.split('.', 1)[1].strip()
                if len(instr) > 0:
                    instructions.append(instr)
            except:
                continue
        else:
            instructions.append(line)
    return instructions


# Step 2: For each instruction, get the ROS command
def generate_ros_command(instruction):
    prompt = f"""You are a robotics assistant. Translate the following instruction into a valid ROS command.
Instruction: {instruction}
ROS Command:"""
    return call_ollama(prompt)


# Step 3: Run pipeline and save to JSONL
instructions = generate_instructions(NUM_TASKS)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for instr in tqdm(instructions, desc="Generating ROS command pairs"):
        try:
            ros_cmd = generate_ros_command(instr)
            f.write(json.dumps({
                "prompt": instr,
                "completion": ros_cmd
            }) + '\n')
        except Exception as e:
            print(f"❌ Error processing '{instr}': {e}")

print(f"\n✅ Generated dataset saved to: {OUTPUT_FILE}")