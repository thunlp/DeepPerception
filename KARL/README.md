# KARL: Knowledge-Aware Reasoning and Reinforcement Learning for Knowledge-Intensive Visual Grounding

Xinyu Ma∗ , Ziyang Ding∗ , Zhicong Luo, Chi Chen4, Zonghao Guo, Xuebo Liu, Derek F. Wong, Zhen Zhao, Xiaoyi Feng, Maosong Sun

<a href='https://deepperception-kvg.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2503.12797'><img src='https://img.shields.io/badge/Paper-PDF-Green'></a> 
<a href='https://huggingface.co/MaxyLee/DeepPerception'><img src='https://img.shields.io/badge/Model-Huggingface-yellow'></a> 
<a href='https://huggingface.co/datasets/MaxyLee/KVG-Bench'><img src='https://img.shields.io/badge/Benchmark-Huggingface-orange'></a> 
<a href='https://huggingface.co/datasets/Oscar-dzy/KARL'><img src='https://img.shields.io/badge/Dataset-Huggingface-purple'></a> 

This is the official repository of **KARL**, an MLLM enhanced with Knowledge-Aware Reinforcement Learning.

## Release

- [x] **`2026.03.28`** 🔥Release the KARL evaluation & training code and model in [`🤗HuggingFace`](https://huggingface.co/MaxyLee/DeepPerception).
- [x] **`2026.03.28`** 🔥KARL Paper has been released in [`📕Arxiv`](https://arxiv.org/abs/2503.12797).

## Overview

<p align="center">
    <img src="figs/header.png" width="100%"></a><br>
    Figure 1: (a) While the MLLM can correctly recognize the entity (Q1), it fails to ground it (Q2), revealing an inconsistency between knowledge and grounding. Our method integrates knowledge-guided reasoning to bridge this gap. (b) (b) \ours achieves substantially stronger grounding performance than the baseline model and zero-shot CoT prompting, showing that knowledge-guided reasoning for KVG cannot be effectively induced by simple prompting alone.
</p>