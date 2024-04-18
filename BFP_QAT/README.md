# Training with BFP quantization in Pytorch

## Requirements
python>=3.6
pytorch>=1.1

## Usage
```bash
python main.py --data_path=. --dataset=IMAGENET --model=ResNet18LP --batch_size=256 --wd=1e-4 --lr_init=0.1 --epochs=100 --tile=32\
--weight-exp=2 --weight-man=5 \
--activate-exp=2 --activate-man=5 \
--error-exp=4 --error-man=3
```

This will train a Resnet-18 model using the 8-bit BFP format(tile size = 32, share exp = 2).


