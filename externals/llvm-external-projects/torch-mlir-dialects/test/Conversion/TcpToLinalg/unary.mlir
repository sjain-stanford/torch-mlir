// RUN: torch-mlir-dialects-opt <%s -convert-tcp-to-linalg -split-input-file | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @tanh(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[BBARG0:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[TANH:.*]] = math.tanh %[[BBARG0]] : f32
// CHECK:           linalg.yield %[[TANH]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @tanh(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.tanh %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @sigmoid(
// CHECK-SAME:                %[[ARG:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:         %[[CONST0:.*]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.*]] = tensor.dim %[[ARG]], %[[CONST0]] : tensor<?x?xf32>
// CHECK:         %[[CONST1:.*]] = arith.constant 1 : index
// CHECK:         %[[DIM1:.*]] = tensor.dim %[[ARG]], %[[CONST1]] : tensor<?x?xf32>
// CHECK:         %[[EMPTY_TENSOR:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]]) : tensor<?x?xf32>
// CHECK:         %[[GENERIC:.*]] = linalg.generic {
// CHECK-SAME:                        indexing_maps = [#[[MAP]], #[[MAP]]],
// CHECK-SAME:                        iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:                        ins(%[[ARG]] :  tensor<?x?xf32>)
// CHECK-SAME:                        outs(%[[EMPTY_TENSOR]] : tensor<?x?xf32>) {
// CHECK:         ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[CONST:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[NEG:.*]] = arith.negf %[[IN]] : f32
// CHECK:           %[[EXP:.*]] = math.exp %[[NEG]] : f32
// CHECK:           %[[ADD:.*]] = arith.addf %[[EXP]], %[[CONST]] : f32
// CHECK:           %[[DIV:.*]] = arith.divf %[[CONST]], %[[ADD]] : f32
// CHECK:           linalg.yield %[[DIV]] : f32
// CHECK:         } -> tensor<?x?xf32>
// CHECK:         return %[[GENERIC]] : tensor<?x?xf32>
// CHECK:       }
func.func @sigmoid(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = tcp.sigmoid %arg0 : tensor<?x?xf32> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
