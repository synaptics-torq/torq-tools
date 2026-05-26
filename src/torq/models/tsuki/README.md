# Tsuki Model SL261x (WIP)

## Getting Started

## Exporting the Model
### TODO: Tsuki export instructions

## Making Tsuki Static

## Making the Model Compatible
### 1. Layer Normalization Decomposition
### 2. Instance Normalization Decomposition
### 3. Constant Folding
### 4. Convert 1x1 Conv to GEMM
### 5. More Normalization Conversions
#### 5.1. ONNX Tiling for ReduceSum and ReduceMean
#### 5.2. Transpose -> ReduceMean(last axis) -> Tranpose back
#### 5.3. Reciprocal -> Mul INTO Div
#### 5.4. POW(x, 2) INTO mul(x,x)
#### 5.5. POW(x, 3) INTO mul(x, mul(x, x))
### 6. Activation Conversion
#### 6.1. LeakyRelu INTO Abs -> Mul -> Mul -> Add
#### 6.2. Rank 4 Softmax INTO Reshape -> Softmax -> Reshape
### 7. Tile large matmul/softmax across attention head axis
### 8. Further tiling of large softmax/attention head across k dimension
### 9. Decompose ConvTranspose using im2col
### 10. Convert ReduceSum(x) INTO ReduceMean(x) * len(x)

## Compiliation Instructions