# Run unit tests.
cmake --build build --target check-torch-mlir

# Run Python regression tests.
cmake --build build --target check-torch-mlir-python
