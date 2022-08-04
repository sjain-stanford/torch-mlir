#!/usr/bin/env bash

torch_binary="${TORCH_BINARY:-ON}"
target_arch="${TARGET_ARCH:-x86_64}"

# Configure cmake to build torch-mlir in-tree
cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_LINKER=lld \
  -DPython3_EXECUTABLE="$(which python)" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS="torch-mlir;torch-mlir-dialects" \
  -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$(pwd)" \
  -DLLVM_EXTERNAL_TORCH_MLIR_DIALECTS_SOURCE_DIR="$(pwd)/externals/llvm-external-projects/torch-mlir-dialects" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DTORCH_MLIR_ENABLE_MHLO=ON \
  -DTORCH_MLIR_USE_INSTALLED_PYTORCH="${torch_binary}" \
  -DCMAKE_OSX_ARCHITECTURES="${target_arch}" \
  -DMACOSX_DEPLOYMENT_TARGET=10.15 \
  -DLLVM_TARGETS_TO_BUILD=host \
  externals/llvm-project/llvm

# Build just torch-mlir (not all of LLVM)
cmake --build build --target tools/torch-mlir/all
