# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

from typing import List

import torch
import torch.nn as nn
from torch.export import Dim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_compiler, get_aot_graph_name, set_model_name

from torch_mlir import fx


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()



@run
# CHECK-LABEL: test_import_frozen_exported_program_with_dynamic_shapes
# CHECK:     func.func @test_net(%[[ARG0:[a-zA-Z0-9]+]]: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,4],f32>
def test_import_frozen_exported_program_with_dynamic_shapes():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            a = torch.tanh(x)
            b = torch.sigmoid(y)
            return torch.cat((a, b))


    x_batch = Dim("x_batch")
    y_batch = Dim("y_batch")
    dynamic_shapes = {"x": {0: x_batch}, "y": {0: y_batch}}
    m = fx.export_and_import(Basic(), torch.randn(3, 4), torch.randn(5, 4), dynamic_shapes=dynamic_shapes, func_name="test_net")
    print(m)

