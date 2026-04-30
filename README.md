# 🍎 FreshCheck AI — Food Quality Inspector

**Advanced Practical Machine Learning — aivancy School**
**Dataset:** Fresh and Stale Food Images (Kaggle)
**Model:** Fine-tuned MobileNetV2 + Grad-CAM visualization

---

## What it does
Upload a photo of any fruit or vegetable → AI instantly classifies
it as **Fresh ✅** or **Rotten ❌** and shows a heatmap of what
the model looked at to make its decision.

---

## Architecture


## Why these choices?

| Decision | Reason |
|---|---|
| MobileNetV2 | Lightweight, fast on CPU, real-world ready |
| Freeze early layers | Low-level features transfer from ImageNet |
| AdamW optimizer | Correct weight decay, better generalization |
| Class weights in loss | Handles fresh/rotten imbalance |
| CosineAnnealingLR | Smooth LR decay, avoids overshooting |
| Grad-CAM | Explains model decisions visually |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
kaggle datasets download sweekar/fresh-and-stale-images-of-fruits-and-vegetables
unzip fresh-and-stale-images-of-fruits-and-vegetables.zip -d ./data/food
```

### 3. Train
```bash
python train.py --data_root ./data/food --epochs 15 --batch_size 32
```

### 4. Run demo
```bash
streamlit run app.py
```

---

## File Structure


---

## Oral Defense Answers

| Question | Answer |
|---|---|
| Why MobileNetV2? | Lightweight, fast, designed for real-world apps |
| Why freeze early layers? | ImageNet features transfer well, only adapt high-level food features |
| Why AdamW? | Weight decay decoupled from gradient — better regularization |
| Why class weights? | Prevents model bias toward majority class |
| What is Grad-CAM? | Backprop class score to last conv layer → weighted feature map = heatmap |

---

## References
- Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.