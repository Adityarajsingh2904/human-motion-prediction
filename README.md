
# 🧍‍♂️ human-motion-prediction – GCN + DCT Based Human Pose Forecasting
![CI](https://github.com/Adityarajsingh2904/human-motion-prediction/actions/workflows/python-ci.yml/badge.svg)

**human-motion-prediction** is a graph-based deep learning model designed to forecast future human motion sequences.  
It leverages **Discrete Cosine Transform (DCT)** and **Multi-Scale Residual Graph Convolutional Networks (MSR-GCN)** to achieve state-of-the-art performance on standard datasets like **Human3.6M** and **CMU Mocap**.

---

## 📌 Key Features

- 📹 Predicts 50 future frames from 25 past frames
- 🧠 Multi-scale residual GCN architecture
- 🦴 Graph-based modeling of human skeletons
- 🏆 Outperforms prior SOTA on Human3.6M & CMU datasets
- 📊 Extensive evaluation and comparison

---

## 🧠 Model Architecture

The model encodes spatiotemporal skeleton features using:
- **DCT transforms** for frequency-domain encoding
- **Multi-scale GCN blocks** for joint dependency modeling
- **Residual connections and intermediate losses** for enhanced supervision

---

## 🗂️ Project Structure

```
human-motion-prediction/
├── main.py               # Entry point for training and testing
├── short_term_main.py    # For short-term prediction
├── model/                # Core GCN architecture
├── data/                 # Data loading and processing scripts
├── README.md             # This file
```

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch ≥ 1.7.0 (CUDA recommended)
- NVIDIA GPU (e.g., RTX 3090)

### Installation

```bash
git clone https://github.com/Adityarajsingh2904/human-motion-prediction.git
cd human-motion-prediction
pip install -r requirements.txt
```

---

## 📥 Datasets

- [Human3.6M](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)
- [CMU Mocap](http://mocap.cs.cmu.edu/)

---

## 🧪 Usage

```bash
python main.py              # Train & evaluate
python short_term_main.py   # Short-term pose prediction
```

---

## 👤 Maintainer

**Aditya Raj Singh**  
📧 thisis.adityarajsingh@gmail.com  
🔗 [GitHub](https://github.com/Adityarajsingh2904)

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more info.
