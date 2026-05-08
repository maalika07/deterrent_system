# Agentic AI Scarecrow - Wildlife Intrusion Deterrent System

An autonomous IoT-based system that uses Agentic AI and Reinforcement Learning to deter wildlife from farmlands, protecting agricultural yields without causing harm to animals.



## Problem Statement

Wildlife raiding farmlands causes 10-15% of total agricultural loss in India annually. Existing deterrent systems rely on static, repetitive patterns that highly intelligent animals such as elephants, boars, and monkeys quickly learn to ignore. This is known as the Habituation Problem. Current systems also lack species-specificity, have no memory of past interactions, and require manual human intervention during dangerous night-time raids.

---

## Overview

The Agentic AI Scarecrow is a self-learning, edge-deployed wildlife deterrent system that solves the Habituation Problem by using Reinforcement Learning to dynamically select the most effective deterrent sound for each detected species. The system identifies animals in real time using computer vision, selects an appropriate acoustic deterrent, learns from the outcome, and improves with every encounter.

The system is deployed on a Raspberry Pi 5 and operates 24 hours a day with no human intervention required.

---

## Key Features

- Real-time wildlife detection using a custom-trained ResNet-50 classifier and YOLOv8n
- Species-specific acoustic deterrence with intelligent randomization to prevent habituation
- Reinforcement Learning brain using an epsilon-greedy Q-Table stored in a persistent JSON file
- Dual perception pipeline with a fast local path and an accurate fallback path via Llama 3.2 Vision
- 20-second deterrent session with a 2-second silence gap between audio repeats
- Long-range farmer alerts via LoRa communication modules
- Fully autonomous with no manual intervention required

---

## System Architecture

The system is organized into four layers.

**Perception Layer** - An OpenCV capture loop reads frames from the USB camera. Frames are preprocessed using TorchVision transforms (resize to 224x224, normalize with ImageNet stats) and passed to the ResNet-50 model. Detections with confidence above 0.85 are forwarded to the decision engine.

**Decision Engine (RL Agent)** - An epsilon-greedy strategy (epsilon = 0.2) queries the Q-Table stored in scarecrow_rl_brain.json to select the optimal deterrent sound for the detected species. If no prior data exists, Q-values default to 5.0.

**Actuation Layer** - Pygame Mixer plays the selected audio file. The session lasts 20 seconds, with a 2-second gap between repeats. Audio files are organized per species in the sounds/ directory.

**Learning Loop** - When the animal leaves the frame, a reward is calculated (Reward = 20 minus active time, or -5 on timeout). The Q-Table is updated using the rule: Q = Q + 0.2 * (reward - Q). The updated brain is saved persistently to JSON.

---

## Dual Perception Pipeline

**Fast Path (ResNet-50 on-device)**
Runs entirely on the Raspberry Pi 5. Suitable for real-time, low-latency inference. Classifies 5 species: Chital, Elephant, Nilgai, Peacock, and Boar.

**Accurate Path (YOLOv8n + Llama 3.2 Vision)**
The Raspberry Pi acts as a scout, running YOLOv8n to detect animals. On detection, it encodes the frame using OpenCV and sends it via HTTP POST to a laptop server running Ollama with Llama 3.2 Vision. The server returns the species name, and the Pi executes the appropriate scare sound. This path is used for higher accuracy or fallback when ResNet confidence is low.

---

## Hardware Stack

| Component | Role |
|---|---|
| Raspberry Pi 5 | Central edge processor |
| USB Camera (V4L2) | Visual input |
| Speaker and Amplifier | Acoustic deterrent output |
| LoRa Module | Long-range farmer alert communication |
| IR Camera (planned) | Night-time detection |

---

## Software Stack

| Component | Purpose |
|---|---|
| Python | Core application language |
| PyTorch and TorchVision | Deep learning inference |
| OpenCV | Video capture and image preprocessing |
| Pygame | Audio playback and mixer control |
| YOLOv8n (Ultralytics) | Real-time animal detection |
| Llama 3.2 Vision via Ollama | Accurate species identification |
| JSON | Persistent RL brain storage |
| Flask | HTTP server for the Llama inference endpoint |






## Challenges

- Night Vision: Standard webcams cannot see in total darkness. Planned fix is to switch to IR cameras.
- Habituation Velocity: Highly intelligent animals may eventually ignore even randomized sounds. Mitigation through continuous RL updates.
- Environmental Noise: Wind and rain may affect audio perception accuracy. Mitigation through confidence thresholding.

---

## References

- YOLOv8 Documentation - https://docs.ultralytics.com
- PyTorch ResNet Documentation - https://pytorch.org/vision/stable/models.html
- Ollama Local LLM Runtime - https://ollama.com
- Pygame Audio Documentation - https://www.pygame.org/docs/ref/mixer.html
- Raspberry Pi 5 Documentation - https://www.raspberrypi.com/documentation/

---

