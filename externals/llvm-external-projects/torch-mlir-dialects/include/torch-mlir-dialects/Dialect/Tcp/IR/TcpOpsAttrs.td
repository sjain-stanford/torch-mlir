//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TCP_OPS_ATTRS
#define TORCH_MLIR_DIALECT_TCP_OPS_ATTRS

include "mlir/IR/AttrTypeBase.td"

def Tcp_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "parseDimSizes($_parser)";
  let printer = "printDimSizes($_printer, $_self)";
}

def Tcp_ConvDimensionNumbers : AttrDef<Tcp_Dialect, "ConvDimensionNumbers"> {
  let cppNamespace = "::mlir::tcp";
  let mnemonic = "conv";
  let summary = "Structure of dimension information for conv op";
  let parameters = (ins
    "int64_t":$inputBatchDimension,
    "int64_t":$inputFeatureDimension,
    Tcp_Dims:$inputSpatialDimensions,
    "int64_t":$kernelInputFeatureDimension,
    "int64_t":$kernelOutputFeatureDimension,
    Tcp_Dims:$kernelSpatialDimensions,
    "int64_t":$outputBatchDimension,
    "int64_t":$outputFeatureDimension,
    Tcp_Dims:$outputSpatialDimensions
  );

  let assemblyFormat = "`<` params `>`";
}


def Tcp_ConvolutionAttributes {
  dag attributes = (ins
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    // Default value: one for each of the spatial dimension.
    OptionalAttr<I64ElementsAttr>:$input_dilation,
    Tcp_ConvDimensionNumbers:$dimension_numbers,
    I64Attr:$feature_groups
  );
}

#endif // TORCH_MLIR_DIALECT_TCP_OPS_ATTRS