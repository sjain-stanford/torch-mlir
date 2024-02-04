import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck
from torch_mlir.jit_ir_importer import ClassAnnotator, ModuleBuilder
from torch_mlir.jit_ir_importer.torchscript_annotations import extract_annotations
from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.annotations import annotate_args, export


def test_convnet_torch_mlir():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(123, 128)
            self.fc2 = nn.Linear(128, 10)
            self.train(False)

        @export
        @annotate_args(
            [
                None,
                (
                    [1, 1, 128, 128],
                    torch.float32,
                    True,
                ),
            ]
        )
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, [1, 1])
            x = torch.flatten(x, 0, 2)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = torch.tanh(x)
            return x

    program = Net()
    scripted = torch.jit.script(program)
    mb = ModuleBuilder()
    class_annotator = ClassAnnotator()
    extract_annotations(
        program, scripted, class_annotator
    )
    mb.import_module(scripted._c, class_annotator)

    with mb.module.context:
        pm = PassManager.parse(
            "builtin.module(torchscript-module-to-torch-backend-pipeline,torch-backend-to-tosa-backend-pipeline)"
        )
        pm.run(mb.module.operation)

    module_str = str(mb.module)

    check_pattern = """
    # CHECK: func @forward(%arg0: tensor<1x1x128x128xf32>) -> tensor<7872x10xf32>
    # CHECK: %0 = "tosa.const"()
    # CHECK: %1 = "tosa.const"()
    # CHECK: %32 = tosa.reshape %30 {new_shape = array<i64: 1, 128, 10>} : (tensor<128x10xf32>) -> tensor<1x128x10xf32>
    # CHECK: %33 = tosa.matmul %31, %32 : (tensor<1x7872x128xf32>, tensor<1x128x10xf32>) -> tensor<1x7872x10xf32>
    # CHECK: %34 = tosa.reshape %33 {new_shape = array<i64: 7872, 10>} : (tensor<1x7872x10xf32>) -> tensor<7872x10xf32>
    # CHECK: %35 = tosa.add %34, %8 : (tensor<7872x10xf32>, tensor<1x10xf32>) -> tensor<7872x10xf32>
    # CHECK: %36 = tosa.tanh %35 : (tensor<7872x10xf32>) -> tensor<7872x10xf32>
    # CHECK: return %36 : tensor<7872x10xf32>
    """
    FileCheck().run(check_pattern, module_str)