# Tsuki Model SL261x (WIP)

## Getting Started

## Exporting the Model
### TODO: Tsuki export instructions

## Making Tsuki Static

## Making the Model Compatible
### 1. Layer/Instance Normalization Decomposition
```
python3 -m src.torq.tools.decompose_norm -i ../models/tsuki/tsuki_static_float32.onnx -o model.onnx
```
### 2. Constant Folding
```
python3 -m src.torq.tools.fold_constants -i <model_in>.onnx -o <model_out>.onnx
```
### 3. Convert 1x1 Conv to GEMM
### 4. More Normalization Conversions
#### 4.1. ONNX Tiling for ReduceSum and ReduceMean
#### 4.2. Transpose -> ReduceMean(last axis) -> Tranpose back
#### 4.3. Reciprocal -> Mul INTO Div
#### 4.4. POW(x, 2) INTO mul(x,x)
#### 4.5. POW(x, 3) INTO mul(x, mul(x, x))
### 5. Activation Conversion
#### 5.1. LeakyRelu INTO Abs -> Mul -> Mul -> Add
#### 5.2. Rank 4 Softmax INTO Reshape -> Softmax -> Reshape
### 6. Tile large matmul/softmax across attention head axis
### 7. Further tiling of large softmax/attention head across k dimension
### 8. Decompose ConvTranspose using im2col
### 9. Convert ReduceSum(x) INTO ReduceMean(x) * len(x)

## Compiliation Instructions