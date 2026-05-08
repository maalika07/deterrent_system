# Model Comparison — ResNet50 vs EfficientNet-B0 vs MobileNetV3-Large

## Wildlife Intrusion Deterrent System — Team D13

---

## 1. Overview

Three CNN architectures were evaluated for real-time animal classification on 9 classes
(8 species + background). All models were fine-tuned on the same dataset of 2,844 images
(316 per class), split 80/20 with random seed 42. Training ran for 15 epochs with
batch size 32, Adam optimizer, and ImageNet pretrained weights.

---

## 2. Full Metrics Comparison

| Metric              | ResNet50       | EfficientNet-B0 | MobileNetV3-Large |
|---------------------|:--------------:|:---------------:|:-----------------:|
| Accuracy            | **0.9965**     | 0.9561          | 0.9666            |
| Balanced Accuracy   | **0.9967**     | 0.9572          | 0.9674            |
| Top-2 Accuracy      | **1.0000**     | 0.9824          | 0.9912            |
| Top-3 Accuracy      | **1.0000**     | 0.9912          | 0.9930            |
| Precision (macro)   | **0.9967**     | 0.9581          | 0.9693            |
| Recall (macro)      | **0.9967**     | 0.9572          | 0.9674            |
| F1 (macro)          | **0.9967**     | 0.9574          | 0.9681            |
| F1 (weighted)       | **0.9965**     | 0.9560          | 0.9667            |
| ROC-AUC (macro)     | **0.9998**     | 0.9978          | 0.9984            |
| MCC                 | **0.9960**     | 0.9506          | 0.9625            |
| Inference (FPS)     | **53**         | 50              | 51                |


> ResNet50 ranks **first across all 11 evaluation metrics** and achieves the highest FPS.

---

## 3. Per-Class Metrics

### ResNet50

| Class          | Precision | Recall | F1-Score | Support |
|----------------|:---------:|:------:|:--------:|:-------:|
| Peacock        | 1.000     | 1.000  | 1.000    | 67      |
| Bonnet Macaque | 1.000     | 1.000  | 1.000    | 67      |
| Chital         | 1.000     | 1.000  | 1.000    | 63      |
| Elephant       | 1.000     | 1.000  | 1.000    | 64      |
| No Animal      | 1.000     | 1.000  | 1.000    | 65      |
| Pig            | 0.985     | 0.985  | 0.985    | 65      |
| Porcupine      | 1.000     | 1.000  | 1.000    | 54      |
| Street Dogs    | 1.000     | 1.000  | 1.000    | 53      |
| Wild Boar      | 0.986     | 0.986  | 0.986    | 71      |

> 7 out of 9 classes achieve **perfect 1.000** scores. Only pig and wild boar fall
> slightly below due to visual similarity between the two species.

---

### EfficientNet-B0

| Class          | Precision | Recall | F1-Score | Support |
|----------------|:---------:|:------:|:--------:|:-------:|
| Peacock        | 0.957     | 0.985  | 0.971    | 67      |
| Bonnet Macaque | 1.000     | 0.970  | 0.985    | 67      |
| Chital         | 0.954     | 0.984  | 0.969    | 63      |
| Elephant       | 1.000     | 0.984  | 0.992    | 64      |
| No Animal      | 0.952     | 0.908  | 0.929    | 65      |
| Pig            | 0.905     | 0.877  | **0.891**| 65      |
| Porcupine      | 0.981     | 0.981  | 0.981    | 54      |
| Street Dogs    | 0.981     | 0.981  | 0.981    | 53      |
| Wild Boar      | 0.893     | 0.944  | **0.918**| 71      |

> Worst performing model overall. Pig F1 of 0.891 is the lowest single-class
> score across all three models. 10.8% of pigs misclassified as wild boar.

---

### MobileNetV3-Large

| Class          | Precision | Recall | F1-Score | Support |
|----------------|:---------:|:------:|:--------:|:-------:|
| Peacock        | 1.000     | 0.970  | 0.985    | 67      |
| Bonnet Macaque | 0.970     | 0.970  | 0.970    | 67      |
| Chital         | 0.969     | 1.000  | 0.984    | 63      |
| Elephant       | 0.985     | 1.000  | 0.992    | 64      |
| No Animal      | 0.984     | 0.954  | 0.969    | 65      |
| Pig            | 0.921     | 0.892  | **0.906**| 65      |
| Porcupine      | 1.000     | 0.981  | 0.991    | 54      |
| Street Dogs    | 1.000     | 0.981  | 0.990    | 53      |
| Wild Boar      | 0.895     | 0.958  | **0.925**| 71      |

> Better than EfficientNet-B0 overall. Chital and elephant achieve perfect recall.
> Pig remains the hardest class (F1: 0.906) — 9.2% misclassified as wild boar.

---

## 4. Confusion Matrix Highlights

| Model             | Pig → Wild Boar errors | Wild Boar → Pig errors | Other notable errors         |
|-------------------|:----------------------:|:----------------------:|------------------------------|
| ResNet50          | 1                      | 1                      | None — all others perfect    |
| EfficientNet-B0   | 7                      | 3                      | No Animal ↔ Chital (2 each)  |
| MobileNetV3-Large | 6                      | 3                      | No Animal ↔ Chital (1 each)  |

> Pig vs Wild Boar is the hardest pair across **all three models** due to
> overlapping body shape, colour, and texture in camera trap images.

---

## 5. Speed vs Accuracy

| Model             | Inference (FPS) | Accuracy | Verdict                          |
|-------------------|:---------------:|:--------:|----------------------------------|
| ResNet50          | **53**          | **99.65%**| Fastest AND most accurate    |
| MobileNetV3-Large | 51              | 96.66%   | 2 FPS slower, 3% less accurate   |
| EfficientNet-B0   | 50              | 95.61%   | Slowest, least accurate          |

> All three models exceed 49 FPS — well above the minimum needed for
> real-time detection (5 FPS). The speed difference between models is
> negligible (3 FPS total range), making accuracy the only deciding factor.

---

## 6. Why EfficientNet-B0 Was Rejected

- Lowest accuracy (95.61%) — nearly **4% below ResNet50**
- Worst F1, MCC, and Cohen Kappa across all metrics
- Pig class F1 of 0.891 — worst single-class score of all three models
- 10.8% pig→wild\_boar confusion — unacceptable for a deterrence system
  (wrong species = wrong sound played)
- No speed advantage despite being a smaller model — 50 FPS vs 53 FPS
- Compound scaling trades accuracy for model size, which is unnecessary
  since Raspberry Pi 5 handles ResNet50 at full speed

---

## 7. Why MobileNetV3-Large Was Rejected

- Accuracy of 96.66% — still **3% below ResNet50**
- Pig class F1 of 0.906 — 9.2% misclassified as wild boar
- Only 2 FPS slower than ResNet50 — no meaningful real-time advantage
- Optimised for mobile on-device inference — an unnecessary trade-off
  since Raspberry Pi 5 runs ResNet50 comfortably in real time
- Depth ablation study showed that truncating layers (MobileNetV3's design
  philosophy) causes significant accuracy drops — L3 truncation lost 4.76%
  in the depth study, mirroring MobileNetV3's behaviour

---

## 8. Why ResNet50 Was Chosen

- **Best accuracy (99.65%)** — leads all 11 evaluation metrics
- **7 out of 9 classes at perfect F1 = 1.000**
- **Fastest inference at 53 FPS** — real-time on Raspberry Pi 5
- **Skip connections** prevent vanishing gradients — stable training
- **Strong ImageNet pretraining** — excellent transfer learning on small datasets
- **Proven architecture** — widely validated for fine-grained classification tasks
- Full sequential backbone confirmed as optimal by all three ablation studies

---

## 9. Ablation Study Impact on Final Choice

| Ablation Study       | Winner                    | Accuracy | Key Finding                                      |
|----------------------|:-------------------------:|:--------:|--------------------------------------------------|
| Head Architecture    | Proposed Fine-tuned Head  | 0.9778   | +0.64% over linear head                         |
| Depth Truncation     | ResNet Full (L4)          | 0.9746   | Removing one block costs 4.76% accuracy          |
| Layer Connections    | Full Sequential           | 0.9714   | Cross-stage bypass loses 6.03% despite 16.9M params |

> All three ablation studies independently confirmed that the **full ResNet50
> with a fine-tuned classification head and standard sequential connections**
> is the optimal configuration. Final deployed accuracy: **99.65% at 53 FPS**.

---

*Dataset: 2,844 images | 9 classes | 316 per class | 80/20 split | Seed 42*
*Training: 15 epochs | Batch 32 | Adam | ImageNet pretrained weights*
*Hardware: Raspberry Pi 5 | USB Camera | Inference measured on-device*