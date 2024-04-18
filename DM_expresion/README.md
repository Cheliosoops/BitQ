# Bitwidth-aware Data Movement Expression and Trade-off Optimization.

## Requirements
python>=3.6
pytorch>=1.1
amplpy, sympy, joblib（lasted）
AMPL -- https://ampl.com/try-ampl/download-a-free-demo/ (minimum version:  20181102)
IPOPT -- https://ampl.com/products/solvers/all-solvers-for-ampl/ (minimum version:3.12.13)

## Usage
```step1 bash
./run_fromfile.sh resnet18
```
```step2 bash
./auto_run.sh resnet18
```

step1: This will run a data movement expression of ResNet-18.
step2: Trade-off optimization.

