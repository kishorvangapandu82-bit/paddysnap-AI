# PADDYSNAP AI — Official Model Performance Comparison Table

Evaluated on the **Test Set (1,562 Images, Seed = 42)**.

| Rank | Model Architecture | Category | Parameters | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Training Time |
|:-----|:-------------------|:---------|:-----------|:-------------|:--------------|:-----------|:-------------|:--------------|
| 1 | **EfficientNetV2-S** | Compound Scaled (ImageNet) | 20.1903 M | **97.38** | **97.34** | **97.44** | **97.35** | **12.6 min** |
| 2 | **ConvNeXt-Tiny** | Modernized CNN (ImageNet) | 27.8278 M | **97.06** | **96.72** | **96.73** | **96.68** | **34.7 min** |
| 3 | **ResNet-50** | Residual Network (ImageNet) | 23.5285 M | **96.99** | **96.66** | **96.99** | **96.81** | **12.6 min** |
| 4 | **DenseNet-121** | Dense Feature Reuse (ImageNet) | **6.9641 M** | **96.93** | **96.29** | **97.38** | **96.80** | **13.1 min** |
| 5 | **Custom PaddySnapNet** | Residual + SE-Attention (Scratch) | 7.8454 M | **93.09** | **91.79** | **93.85** | **92.74** | **14.1 min** |
