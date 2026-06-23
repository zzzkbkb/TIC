## The official implementation of "Resource-Efficient Learned Image Compression via Asymmetric Cross-Band Modeling"
---
<img src="images/Overall architecture.png" width="1200">

## Repository Structure
---
```bash
.
├── compressai/models
│   ├── __init__.py
│   ├── tic_light.py          # main model definition (mambalic_model)
├── train.py                 # training script
├── test.py                  # evaluation script
```

## Training
---
If you want to train the network on the Flicker2W dataset, please use the following instructions: Single gpu for train:

```bash
python train.py --model tic_light --dataset /d/Downloads/flicker2W --epoch 500 --seed 42 --batch-size 16 --N 128 --M 192 --cuda
```

Testing on Kodak24
--
```bash
python test.py checkpoint d/Downloads/Kodak24 -a tic_light  --cuda -v
```
