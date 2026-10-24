# FedGuard-IoT: Privacy-Preserving Federated Learning

## 📖 Project Overview
This project implements a simulated privacy-preserving Federated Learning (FL) framework for Internet of Things (IoT) sensor networks. It is designed to mitigate the privacy, security, and communication overhead challenges inherent in traditional centralized machine learning architectures.

By utilizing **Federated Averaging (FedAvg)** combined with a **Differential Privacy Layer (Gradient Clipping + Gaussian Noise)**, this system allows distributed IoT edge nodes to collaboratively train an Intrusion Detection System (IDS) without ever sharing their raw sensor logs.

## 🏗️ System Architecture
The framework is divided into 5 distinct layers:
1. **Distributed IoT Data Partitioning Layer:** Simulates independent IoT nodes using Non-IID slices of the Edge-IIoTset dataset.
2. **Client-Side Local Model Training:** PyTorch-based neural networks training locally via Stochastic Gradient Descent (SGD).
3. **Local Privacy Protection Layer:** Secures model updates against inference attacks using controlled gradient clipping and noise injection ($N(0, \sigma^2C^2)$).
4. **Protected Update & Aggregation:** Central server orchestrates client updates using the Flower (`flwr`) framework.
5. **Evaluation & Visualization Layer:** Benchmarks accuracy, loss, and privacy-utility trade-offs against a centralized baseline.

## 🗺️ Project Roadmap (The Master Plan)
- [x] **Phase 0:** Dataset Preparation (Ingestion, Cleaning, Non-IID Partitioning).
- [ ] **Phase 1:** Centralized Baseline Model (PyTorch Neural Network for performance benchmarking).
- [ ] **Phase 2:** Standard Federated Learning (Flower Server & Client orchestration without privacy noise).
- [ ] **Phase 3:** The Privacy Layer (Implementing Clipping + Gaussian Noise on client updates).
- [ ] **Phase 4:** Evaluation & Trade-off Analysis (Comparing Centralized vs. FedAvg vs. Private-FedAvg).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning:** PyTorch
* **Federated Learning:** Flower (`flwr`)
* **Data Manipulation:** Pandas, Scikit-Learn, NumPy
* **Dataset:** Edge-IIoTset (IoT Cyber Security Logs)