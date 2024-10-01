
# 🕺 PoseFutureNet – Human Motion Prediction with GCN & DCT

**PoseFutureNet** is a graph-based deep learning model designed to predict future human motion sequences.  
It integrates **Discrete Cosine Transform (DCT)** and **Multi-Scale Residual Graph Convolutional Networks (MSR-GCN)** to deliver state-of-the-art performance on benchmark datasets like **Human3.6M** and **CMU Mocap**.

---

## 📌 Key Features

- 🔁 Predicts 50 future frames from 25 past frames
- ⚙️ Multi-scale residual learning architecture
- 🌐 Graph-based modeling of human skeletons
- 🎯 Outperforms prior SOTA on Human3.6M & CMU datasets
- 🧪 Extensive evaluation and comparison

---

## 🧠 Model Architecture

The model encodes spatiotemporal skeleton features using:
- **DCT transforms** for frequency-domain encoding
- **Multi-scale GCN blocks** for joint dependency modeling
- **Residual connections** and **intermediate losses** for enhanced supervision

---

## 🗂️ Project Structure

```
human-motion-prediction/
├── main.py               # Training and evaluation entry
├── short_term_main.py    # For shorter prediction tasks
├── model/                # GCN and residual architecture
├── data/                 # Dataset preprocessing and loaders
├── README.md             # Project overview
```

---

## 🔧 Setup & Dependencies

### ✅ Requirements

- Python 3.8+
- PyTorch ≥ 1.7.0 (with CUDA)
- GPU: NVIDIA RTX 3090 recommended

### 🛠 Installation

```bash
git clone https://github.com/Adityarajsingh2904/PoseFutureNet.git
cd PoseFutureNet
pip install -r requirements.txt
```

---

## 📥 Dataset

Download Human3.6M and CMU datasets:

- [Human3.6M](http://vision.imar.ro/human3.6m/description.php) – [H3.6M ZIP](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip)
- [CMU Mocap](http://mocap.cs.cmu.edu/)

---

## 🧪 Training & Testing

```bash
python main.py              # Full model training
python short_term_main.py   # Short-term prediction evaluation
```

---

## 👨‍💻 Maintainer

**Aditya Raj Singh**  
📧 thisis.adityarajsingh@gmail.com  
🔗 [GitHub](https://github.com/Adityarajsingh2904)

---

## 📜 License

This project is released under the MIT License.

---
