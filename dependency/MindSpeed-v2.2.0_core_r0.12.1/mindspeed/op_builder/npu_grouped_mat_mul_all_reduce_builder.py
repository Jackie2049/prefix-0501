import torch
from mindspeed.op_builder.builder import MindSpeedOpBuilder


class GroupedMatMulAllReduceOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_grouped_mat_mul_all_reduce"
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def __init__(self):
        super(GroupedMatMulAllReduceOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_grouped_mat_mul_all_reduce.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        if self.TORCH_MAJOR >= 2 and self.TORCH_MINOR >= 1:
            cpp_std = " -std=c++17"
            compile_maroc = " -D__TORCH_2__"
        else:
            cpp_std = " -std=c++14"
            compile_maroc = " -D__TORCH_1__"
        args.append(cpp_std)
        args.append(compile_maroc)
        
        return args
