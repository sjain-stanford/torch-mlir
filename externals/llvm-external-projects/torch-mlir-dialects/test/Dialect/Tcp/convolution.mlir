// RUN: torch-mlir-dialects-opt %s -split-input-file -verify-diagnostics | FileCheck %s


// CHECK-LABEL: func.func @test_convolution(
func.func @test_convolution(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = tcp.convolution %arg0, %arg1 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}