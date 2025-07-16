RoboInstruct: Fine-Tuning LLaMA 3 for Robotic Task Instructions
Project Overview
RoboInstruct fine-tunes Meta AI’s LLaMA 3 (8B) to generate ROS-compatible task instructions for industrial robots, such as pick-and-place or welding tasks in automotive assembly. Leveraging a high-end GPU (NVIDIA RTX 4090), this project integrates advanced NLP with robotics, showcasing skills in AI, automation, and data science.
Dataset

Source: Synthetic dataset of task descriptions and JSON-formatted ROS instructions.
Task: Generate structured robotic task instructions from natural language inputs.
Size: 8,000 training samples, 1,000 evaluation samples.

Requirements

OS: Windows 10/11 with WSL2 (Ubuntu 20.04) for ROS Noetic.
Hardware: NVIDIA RTX 4090, 96GB RAM.
Software: Python 3.10+, PyCharm, CUDA 12.1, cuDNN.
Libraries:pip install torch==2.3.0 transformers==4.45.0 datasets==2.21.0 peft==0.12.0 bitsandbytes==0.43.3 roslibpy==1.3.0



Setup Instructions

Install WSL2 and ROS Noetic:wsl --install
wsl --set-default-version 2

In WSL2 (Ubuntu 20.04):sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc


Set Up PyCharm:
Create a project in PyCharm.
Configure a virtual environment with Python 3.10+.
Install dependencies (see above).
Optionally, set WSL2 as the interpreter for ROS scripts (File > Settings > Project > Python Interpreter > WSL).


Clone the Repository:git clone
cd RoboInstruct


Run the Fine-Tuning Script:python src/finetune_llama3.py


Start ROS Master in WSL2:wsl
source /opt/ros/noetic/setup.bash
roscore


Test ROS Integration:python src/ros_integration.py


Outputs saved in ./finetuned_RoboInstruct.

Results

Accuracy: ~90% (varies based on dataset quality).
F1-Score: ~0.90 (weighted average).
Training logs in ./logs.

Project Structure

src/finetune_llama3.py: Fine-tuning script for LLaMA 3.
src/ros_integration.py: ROS integration script.
data/: Dataset storage (e.g., dataset.json).
results/: Training outputs.
logs/: Training logs.
finetuned_RoboInstruct/: Saved model and tokenizer.

Future Improvements

Expand dataset with diverse industrial tasks.
Optimize LoRA parameters for higher accuracy.
Develop a Streamlit interface for task input.

Skills Demonstrated

Advanced NLP with LLaMA 3
ROS integration for industrial robotics
Python, PyTorch, Hugging Face ecosystem
GPU-accelerated training (NVIDIA RTX 4090)
JSON parsing and automation

Contact

Email: Deepakjeganathan@gmail.com
LinkedIn: www.linkedin.com/in/deepak-jeganathan