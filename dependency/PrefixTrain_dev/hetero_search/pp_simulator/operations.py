from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class BasicConfig:
    p: int  # pp stage count
    m: int  # microbatch count
    v: int  # virtual pipeline count
    c: int  # data copy time
    overlap_c: bool  # overlap data transfer with compute

    @staticmethod
    def is_first_stage(i):
        return i == 0

    def is_last_stage(self, i):
        return i == self.p - 1

    @staticmethod
    def is_first_chunk(i):
        return i == 0

    def is_last_chunk(self, i):
        return i == self.v - 1


@dataclass
class Config(BasicConfig):
    f: int  # forward time
    b: int  # backward time
    a: int  # activation transfer time
    n: Optional[int] = None

    def __post_init__(self):
        if self.n is None:
            self.n = self.p


@dataclass
class HyperConfig(BasicConfig):
    f: list[Union[int, list[int]]]
    b: list[Union[int, list[int]]]
    a: list[list[int]]  # activation send time between different node
    r: list[list[bool]]  # relation between each node True use hccl False use tcp need data copy
    n: Optional[int] = None

    def __post_init__(self):
        if self.n is None:
            self.n = self.p


class Operation:
    op_name = "Operation"

    def __init__(
            self,
            config: Config,
            pp_idx: int,
            batch_idx: int,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        self.config = config
        self.pp_idx = pp_idx
        self.batch_idx = batch_idx
        self.start_time = start_time
        self.pre_operation = pre_operation
        self.end_time = -1

        self.chunk_idx = batch_idx // config.n % config.v
        self.real_batch_idx = config.n * (batch_idx // (config.n * config.v)) + batch_idx % config.n
        self.duration = -1

        self.followed_ops = []
        if self.pre_operation is not None:
            self.pre_operation.add_followed_op(self)

    def is_started(self):
        return self.start_time >= 0

    def update_end_time(self):
        assert self.duration >= 0
        self.end_time = self.start_time + self.duration

    def add_followed_op(self, op: "Operation"):
        self.followed_ops.append(op)

    def start_followed_ops(self):
        assert self.end_time >= 0
        for op in self.followed_ops:
            op.start_time = self.end_time

    def export_trace(self, simple: bool = True):
        return {
            "tid": f"stage{self.pp_idx:02d}",
            "pid": "Main",
            "name": f"{self.op_name}-{self.chunk_idx}-{self.real_batch_idx:02d}",
            "ph": "X",
            "ts": self.start_time * 1000,
            "dur": (self.end_time - self.start_time) * 1000,
        }


class FWD(Operation):
    op_name = "FWD"

    def __init__(
            self,
            config,
            pp_idx: int,
            batch_idx: int,
            vp_idx: int = 0,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
    
        if config.__class__.__name__ == 'Config':
            self.duration = config.f
            if not config.overlap_c:
                if config.is_first_stage(pp_idx) and config.is_first_chunk(self.chunk_idx):
                    self.duration += config.c
                elif config.is_last_stage(pp_idx) and config.is_last_chunk(self.chunk_idx):
                    self.duration += config.c
                else:
                    self.duration += 2 * config.c
        elif config.__class__.__name__ == 'HyperConfig':
            self.duration = config.f[pp_idx]
            if isinstance(self.duration, list):
                self.duration = self.duration[vp_idx]
            if not config.overlap_c:
                if config.is_first_stage(pp_idx) and config.is_first_chunk(self.chunk_idx):
                    if not config.r[pp_idx][(pp_idx + 1) % config.p]:
                        self.duration += config.c
                elif config.is_last_stage(pp_idx) and config.is_last_chunk(self.chunk_idx):
                    if not config.r[pp_idx][(pp_idx - 1) % config.p]:
                        self.duration += config.c
                else:
                    if not config.r[pp_idx][(pp_idx + 1) % config.p]:
                        self.duration += config.c
                    if not config.r[pp_idx][(pp_idx - 1) % config.p]:
                        self.duration += config.c

    def export_trace(self, simple: bool = True):
        cnames = ["rail_response", "thread_state_runnable"]
        if simple:
            return {
                "tid": f"stage{self.pp_idx:02d}",
                "pid": "Main",
                "name": f"F-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }
        else:
            return {
                "tid": "Compute",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"F-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }

class BWD(Operation):
    op_name = "BWD"

    def __init__(
            self,
            config,
            pp_idx: int,
            batch_idx: int,
            vp_idx: int = 0,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
        self.chunk_idx = config.v - batch_idx // config.n % config.v - 1

        if config.__class__.__name__ == 'Config':
            self.duration = config.b
            if not config.overlap_c:
                if config.is_last_stage(pp_idx) and config.is_last_chunk(self.chunk_idx):
                    self.duration += config.c
                elif config.is_first_stage(pp_idx) and config.is_first_chunk(self.chunk_idx):
                    self.duration += config.c
                else:
                    self.duration += 2 * config.c
        elif config.__class__.__name__ == 'HyperConfig':
            self.duration = config.b[self.pp_idx]
            if isinstance(self.duration, list):
                self.duration = self.duration[vp_idx]
            if not config.overlap_c:
                if config.is_last_stage(pp_idx) and config.is_last_chunk(self.chunk_idx):
                    if not config.r[pp_idx][(pp_idx - 1) % config.p]:
                        self.duration += config.c
                elif config.is_first_stage(pp_idx) and config.is_first_chunk(self.chunk_idx):
                    if not config.r[pp_idx][(pp_idx + 1) % config.p]:
                        self.duration += config.c
                else:
                    if not config.r[pp_idx][(pp_idx + 1) % config.p]:
                        self.duration += config.c
                    if not config.r[pp_idx][(pp_idx - 1) % config.p]:
                        self.duration += config.c

    def export_trace(self, simple: bool = True):
        cnames = ["rail_load", "cq_build_passed"]
        if simple:
            return {
                "tid": f"stage{self.pp_idx:02d}",
                "pid": "Main",
                "name": f"B-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }
        else:
            return {
                "tid": "Compute",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"B-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }


class BWD_X(BWD):
    op_name = "BWD_X"
    def export_trace(self, simple: bool = True):
        cnames = ["startup", "startup"]
        if simple:
            return {
                "tid": f"stage{self.pp_idx:02d}",
                "pid": "Main",
                "name": f"BX-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }
        else:
            return {
                "tid": "Compute",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"BX-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": cnames[self.chunk_idx % 2],
            }

class SendFWD(Operation):
    op_name = "SendFWD"

    def __init__(
            self,
            config,
            pp_idx: int,
            batch_idx: int,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
        self.dst_pp_idx = (pp_idx + 1) % config.p

        if config.__class__.__name__ == 'Config':
            self.duration = config.a
            if config.overlap_c:
                self.duration += 2 * config.c
        elif config.__class__.__name__ == 'HyperConfig':
            self.duration = config.a[pp_idx][self.dst_pp_idx]
            if not config.r[pp_idx][self.dst_pp_idx]:
                if config.overlap_c:
                    self.duration += 2 * config.c

    def export_trace(self, simple: bool = True):
        if simple:
            return {}
        else:
            return {
                "tid": "SendFWD",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"S-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": "light_memory_dump",
            }


class RecvFWD(Operation):
    op_name = "RecvFWD"

    def __init__(
            self,
            config: Config,
            pp_idx: int,
            batch_idx: int,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
        self.src_pp_idx = (pp_idx - 1) % config.p
        # self.duration = config.a
        self.pair_send_queue = "SendFWD"
        if self.config.v > 1:
            self.pair_send_idx = batch_idx if not config.is_first_stage(pp_idx) else batch_idx - config.n
        else:
            self.pair_send_idx = batch_idx

    def export_trace(self, simple: bool = True):
        if simple:
            return {}
        else:
            return {
                "tid": "RecvFWD",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"R-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "I",
                "ts": self.start_time * 1000,
            }


class SendBWD(Operation):
    op_name = "SendBWD"

    def __init__(
            self,
            config,
            pp_idx: int,
            batch_idx: int,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
        self.dst_pp_idx = (pp_idx - 1) % config.p
        self.chunk_idx = config.v - batch_idx // config.n % config.v - 1

        if config.__class__.__name__ == 'Config':
            self.duration = config.a
            if config.overlap_c:
                self.duration += 2 * config.c
        elif config.__class__.__name__ == 'HyperConfig':
            self.duration = config.a[pp_idx][self.dst_pp_idx]
            if not config.r[pp_idx][self.dst_pp_idx]:
                if config.overlap_c:
                    self.duration += 2 * config.c

    def export_trace(self, simple: bool = True):
        if simple:
            return {}
        else:
            return {
                "tid": "SendBWD",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"S-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "X",
                "ts": self.start_time * 1000,
                "dur": (self.end_time - self.start_time) * 1000,
                "cname": "thread_state_runnable",
            }


class RecvBWD(Operation):
    op_name = "RecvBWD"

    def __init__(
            self,
            config,
            pp_idx: int,
            batch_idx: int,
            start_time: int = -1,
            pre_operation: Optional["Operation"] = None,
    ):
        super().__init__(config, pp_idx, batch_idx, start_time, pre_operation)
        self.src_pp_idx = (pp_idx + 1) % config.p
        self.chunk_idx = config.v - batch_idx // config.n % config.v - 1

        self.pair_send_queue = "SendBWD"
        if self.config.v > 1:
            self.pair_send_idx = batch_idx if not config.is_last_stage(pp_idx) else batch_idx - config.n
        else:
            self.pair_send_idx = batch_idx

    def export_trace(self, simple: bool = True):
        if simple:
            return {}
        else:
            return {
                "tid": "RecvBWD",
                "pid": f"Stage{self.pp_idx:02d}",
                "name": f"R-{self.chunk_idx}-{self.real_batch_idx:02d}",
                "ph": "I",
                "ts": self.start_time * 1000,
            }
