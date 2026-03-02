# SpikingDiffusion-Neuromorphic-Generative-Modeling
This project explores the integration of **Spiking Neural Networks (SNNs)** with **diffusion-style generative modeling** for images and audio signals. The goal is to investigate whether spike-based neural systems can learn to represent and reconstruct data, and partially reverse stochastic corruption processes.

---

## Project Overview

This repository implements a **toy diffusion-style framework** using SNNs, along with supporting experiments in image reconstruction and spike-based signal encoding.

The project is divided into three main components:

1. **Spiking Autoencoder (MNIST)**
2. **Spiking Diffusion Denoising Model**
3. **Spike-based Audio Rate Coding**

The emphasis is on:
- Understanding **spike-based representations**
- Exploring **denoising as a generative principle**
- Evaluating feasibility of **neuromorphic generative modeling**

---

## 📂 Repository Structure

```
SpikingDiffusion/
│
├── core/
│   ├── models.py
│   ├── diffusion.py
│   ├── utils.py
│
├── 1_spiking_autoencoder_mnist.ipynb
├── 2_spiking_diffusion_denoising_mnist.ipynb
├── 3_spike_based_audio_rate_coding.ipynb
│
└── README.md
```


---

## Methods and Concepts

### 🔹 Spiking Neural Networks (SNNs)
The models are implemented using **Leaky Integrate-and-Fire (LIF)** neurons via `snnTorch`, enabling temporal spike-based computation.

### 🔹 Rate Coding
Static inputs (images and signals) are converted into spike trains using **rate coding**, where intensity is represented by firing frequency.

### 🔹 Diffusion-style Modeling
A simplified diffusion formulation is used:

x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

- Forward process: progressively corrupt data with noise  
- Reverse process: train a model to **predict noise** (denoising)

### 🔹 Poisson Spike Noise
Noise is modeled using a **Poisson-like process**, inspired by biological spike generation.

---

## Notebooks

### 1️⃣ Spiking Autoencoder (MNIST)

- Encodes images into spike trains using rate coding  
- Reconstructs images from spike activity  
- Demonstrates that SNNs can learn **representation and reconstruction**

**Key idea:**  
Spike rates encode information, and temporal averaging recovers signals.

---

### 2️⃣ Spiking Diffusion Denoising (MNIST)

- Implements a **toy diffusion-style corruption process**  
- Trains a spiking neural network to **predict noise**  
- Demonstrates **denoising capability**

#### Key Result

Quantitative evaluation shows:
MSE(Noisy vs Original) > MSE(Denoised vs Original)

This confirms that the model learns to **partially reverse corruption**, validating the diffusion objective.

> Note: Full high-quality generation from pure noise is computationally expensive, especially with SNNs. This work focuses on validating the **denoising component**.
---

### 3️⃣ Spike-based Audio Rate Coding

- Encodes a synthetic waveform into spikes  
- Reconstructs the signal using spike rates  
- Demonstrates that spike-based representations preserve **temporal signal structure**

---

## 📊 Evaluation

The primary evaluation is based on **Mean Squared Error (MSE)**:

- Compare noisy input vs original  
- Compare denoised output vs original  

A lower MSE after denoising indicates successful learning of the reverse diffusion process.

---

## ⚠️ Limitations

- Diffusion models are computationally expensive  
- Training SNNs with surrogate gradients is challenging  
- The current model is a **proof-of-concept**, not a fully converged generative model  
- Generated samples may remain noisy without extensive training

---

## 🚀 Key Takeaways

- SNNs can encode and reconstruct signals using spike-based representations  
- Diffusion-style denoising can be implemented with spiking neurons  
- Quantitative results show meaningful **denoising behavior**  
- This project serves as a **foundation for neuromorphic generative modeling**

---

## 🛠️ Requirements

```bash
pip install torch torchvision snntorch matplotlib tqdm
```
## Acknowledgement
This project is an exploratory research prototype aimed at understanding the intersection of:

-Neuromorphic computing

-Generative modeling

-Spike-based information processing
---
