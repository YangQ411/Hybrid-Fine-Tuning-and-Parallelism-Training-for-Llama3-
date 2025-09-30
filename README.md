# Hybrid Fine-Tuning and Parallelism Training for Llama3

This repository contains the implementation, experiments, and report for the project:

**Hybrid Fine-Tuning and Parallelism for Improving Token Generation Performance in LLaMA3.2-1B**  
Course: CSYE7105 - Parallel Machine Learning & AI (Spring 2025)  
Team: Yang Qu, Rui Wu  

## 🚀 Project Overview
We investigate **efficient fine-tuning** of the LLaMA3.2-1B model using:
- **Low-Rank Adaptation (LoRA)**: Injecting lightweight trainable matrices into attention layers (q_proj, v_proj, o_proj).
- **Mixed Precision Training (FP16 / BF16)**: Improving speed and memory efficiency.
- **Distributed Training (DDP & FSDP)**: Enhancing scalability across multi-GPU HPC clusters.
- **Dask + SLURM Tokenization**: Reducing preprocessing time (80 → 36 minutes for OpenOrca dataset).
- **Gradio UI**: Deploying a chatbot interface mimicking ChatGPT.

---

## 📂 Repository Structure