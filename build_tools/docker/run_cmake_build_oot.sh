#!/usr/bin/env bash

torch_binary="${TORCH_BINARY:-ON}"
target_arch="${TARGET_ARCH:-x86_64}"

# Configure cmake to build torch-mlir out-of-tree
cmake -GNinja -Bllvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DPython3_EXECUTABLE="$(which python)" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm

# Build LLVM
cmake --build llvm-build

cmake -GNinja -Bbuild \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DPython3_EXECUTABLE="$(which python)" \
  -DMLIR_DIR="$(pwd)/llvm-build/lib/cmake/mlir/" \
  -DLLVM_DIR="$(pwd)/llvm-build/lib/cmake/llvm/" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DTORCH_MLIR_ENABLE_MHLO=ON \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH="${torch_binary}" \
  .

# Build torch-mlir
cmake --build build
