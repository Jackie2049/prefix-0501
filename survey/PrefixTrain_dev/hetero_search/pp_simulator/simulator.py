import json

from operations import BWD, FWD, BWD_X, Operation, RecvBWD, RecvFWD, SendBWD, SendFWD

COMPUTE = "Compute"
SENDFWD = "SendFWD"
RECVFWD = "RecvFWD"
SENDBWD = "SendBWD"
RECVBWD = "RecvBWD"


class OperationGenerator:
    def __init__(self, config) -> None:
        self.config = config
        self.operations = {
            i: {
                COMPUTE: [],
                SENDFWD: [],
                RECVFWD: [],
                SENDBWD: [],
                RECVBWD: [],
            }
            for i in range(config.p)
        }

    def generate(self):
        raise NotImplementedError()

    def is_first_stage(self, i):
        return i == 0

    def is_last_stage(self, i):
        return i == self.config.p - 1

    def fwd(self, pp_idx, batch_idx, pre_operation):
        fwd = FWD(self.config, pp_idx, batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][COMPUTE].append(fwd)
        return fwd

    def bwd(self, pp_idx, batch_idx, pre_operation):
        bwd = BWD(self.config, pp_idx, batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][COMPUTE].append(bwd)
        return bwd


class OneFOneBGenerator(OperationGenerator):

    def total_num_microbatches(self):
        return self.config.m

    def num_warmup_microbatches(self, pp_idx):
        return self.config.p - pp_idx - 1

    def generate(self):
        total_num_microbatches = self.total_num_microbatches()
        for i in range(self.config.p):
            num_warmup_microbatches = self.num_warmup_microbatches(i)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            self.recv_fwd(pp_idx=i, batch_idx=0, start_time=0)

            fwd = None
            for j in range(num_warmup_microbatches):
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=j + 1, pre_operation=fwd)
                fwd = self.fwd(i, j, self.operations[i][RECVFWD][-2])
                self.send_fwd(pp_idx=i, batch_idx=j, pre_operation=fwd)

            if num_warmup_microbatches == 0:
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=num_warmup_microbatches)

            bwd = fwd
            for j in range(num_microbatches_remaining):
                fwd_j = j + num_warmup_microbatches
                bwd_j = j

                recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                fwd = self.fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=recv_fwd)
                self.send_fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=fwd)
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=fwd_j + 1, pre_operation=fwd)
                bwd = self.bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=recv_bwd)
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)

            # insert recv_bwd into last fwd bwd
            recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=num_microbatches_remaining, pre_operation=fwd)

            for j in range(num_warmup_microbatches):
                bwd_j = j + num_microbatches_remaining
                recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)
                bwd = self.bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=self.operations[i][RECVBWD][-2])
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)

        return self.operations

    def recv_fwd(self, pp_idx, batch_idx, start_time=-1, pre_operation=None):
        recv_fwd = None
        if not self.is_first_stage(pp_idx):
            recv_fwd = RecvFWD(
                config=self.config,
                pp_idx=pp_idx,
                batch_idx=batch_idx,
                start_time=start_time,
                pre_operation=pre_operation,
            )
        self.operations[pp_idx][RECVFWD].append(recv_fwd)
        return recv_fwd

    def send_fwd(self, pp_idx, batch_idx, pre_operation):
        send_fwd = None
        if not self.is_last_stage(pp_idx):
            send_fwd = SendFWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][SENDFWD].append(send_fwd)
        return send_fwd

    def recv_bwd(self, pp_idx, batch_idx, pre_operation=None):
        recv_bwd = None
        if not self.is_last_stage(pp_idx):
            recv_bwd = RecvBWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][RECVBWD].append(recv_bwd)
        return recv_bwd

    def send_bwd(self, pp_idx, batch_idx, pre_operation=None):
        send_bwd = None
        if not self.is_first_stage(pp_idx):
            send_bwd = SendBWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][SENDBWD].append(send_bwd)
        return send_bwd


class EagerOneFOneBGenerator(OneFOneBGenerator):
    def num_warmup_microbatches(self, pp_idx):
        return 2 * (self.config.p - pp_idx - 1)


class InterleavedOneFOneBGenerator(OneFOneBGenerator):
    def generate_n(self):
        total_num_microbatches = self.total_num_microbatches()
        for i in range(self.config.p):
            num_warmup_microbatches = self.num_warmup_microbatches(i)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            fwd = None
            bwd = None
            self.recv_fwd(pp_idx=i, batch_idx=0, start_time=0)
            for j in range(num_warmup_microbatches):
                fwd = self.fwd(i, j, self.operations[i][RECVFWD][-1])
                self.send_fwd(pp_idx=i, batch_idx=j, pre_operation=fwd)
                self.recv_fwd(pp_idx=i, batch_idx=j + 1, pre_operation=fwd)

            self.recv_bwd(pp_idx=i, batch_idx=0, pre_operation=fwd)
            for j in range(num_microbatches_remaining):
                fwd_j = j + num_warmup_microbatches
                bwd_j = j

                fwd = self.fwd(i, fwd_j, self.operations[i][RECVFWD][-1])
                self.send_fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=fwd)
                self.recv_fwd(pp_idx=i, batch_idx=fwd_j + 1, pre_operation=fwd)

                bwd = self.bwd(i, bwd_j, self.operations[i][RECVBWD][-1])
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)

            for j in range(num_warmup_microbatches):
                bwd_j = j + num_microbatches_remaining
                bwd = self.bwd(i, bwd_j, self.operations[i][RECVBWD][-1])
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)

        return self.operations

    def fwd(self, pp_idx, batch_idx, pre_operation):
        vp_idx = self.get_chunk_idx(batch_idx, True)
        fwd = FWD(self.config, pp_idx, batch_idx, vp_idx, pre_operation=pre_operation)
        self.operations[pp_idx][COMPUTE].append(fwd)
        return fwd

    def bwd(self, pp_idx, batch_idx, pre_operation):
        vp_idx = self.get_chunk_idx(batch_idx, False)
        bwd = BWD(self.config, pp_idx, batch_idx, vp_idx, pre_operation=pre_operation)
        self.operations[pp_idx][COMPUTE].append(bwd)
        return bwd

    def num_warmup_microbatches(self, pp_idx):
        res = (self.config.p - pp_idx - 1) * 2
        res += (self.config.v - 1) * self.config.n
        return res

    def total_num_microbatches(self):
        return self.config.m * self.config.v

    def get_chunk_idx(self, batch_idx: int, is_fwd: bool = True):
        if is_fwd:
            return batch_idx // self.config.n % self.config.v
        else:
            return self.config.v - batch_idx // self.config.n % self.config.v - 1

    def is_first_chunk(self, i):
        return i == 0

    def is_last_chunk(self, i):
        return i == self.config.v - 1

    def recv_fwd(self, pp_idx, batch_idx, start_time=-1, pre_operation=None):
        recv_fwd = None
        chunk_idx = self.get_chunk_idx(batch_idx)
        if not (self.is_first_stage(pp_idx) and self.is_first_chunk(chunk_idx)):
            recv_fwd = RecvFWD(
                config=self.config,
                pp_idx=pp_idx,
                batch_idx=batch_idx,
                start_time=start_time,
                pre_operation=pre_operation,
            )
        self.operations[pp_idx][RECVFWD].append(recv_fwd)
        return recv_fwd

    def send_fwd(self, pp_idx, batch_idx, pre_operation):
        send_fwd = None
        chunk_idx = self.get_chunk_idx(batch_idx)
        if not (self.is_last_stage(pp_idx) and self.is_last_chunk(chunk_idx)):
            send_fwd = SendFWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][SENDFWD].append(send_fwd)
        return send_fwd

    def recv_bwd(self, pp_idx, batch_idx, pre_operation=None):
        recv_bwd = None
        chunk_idx = self.get_chunk_idx(batch_idx, is_fwd=False)
        if not (self.is_last_stage(pp_idx) and self.is_last_chunk(chunk_idx)):
            recv_bwd = RecvBWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][RECVBWD].append(recv_bwd)
        return recv_bwd

    def send_bwd(self, pp_idx, batch_idx, pre_operation=None):
        send_bwd = None
        chunk_idx = self.get_chunk_idx(batch_idx, is_fwd=False)
        if not (self.is_first_stage(pp_idx) and self.is_first_chunk(chunk_idx)):
            send_bwd = SendBWD(config=self.config, pp_idx=pp_idx, batch_idx=batch_idx, pre_operation=pre_operation)
        self.operations[pp_idx][SENDBWD].append(send_bwd)
        return send_bwd


class NanoPipeGenerator(InterleavedOneFOneBGenerator):
    def generate(self):
        total_num_microbatches = self.total_num_microbatches()
        for i in range(self.config.p):
            num_warmup_microbatches = self.num_warmup_microbatches(i)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            self.recv_fwd(pp_idx=i, batch_idx=0, start_time=0)

            fwd = None
            for j in range(num_warmup_microbatches):
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=j + 1, pre_operation=fwd)
                fwd = self.fwd(i, j, self.operations[i][RECVFWD][-2])
                self.send_fwd(pp_idx=i, batch_idx=j, pre_operation=fwd)

            if num_warmup_microbatches == 0:
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=num_warmup_microbatches)

            bwd = fwd
            for j in range(num_microbatches_remaining):
                fwd_j = j + num_warmup_microbatches
                bwd_j = j

                recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                fwd = self.fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=recv_fwd)
                self.send_fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=fwd)
                recv_fwd = self.recv_fwd(pp_idx=i, batch_idx=fwd_j + 1, pre_operation=fwd)
                bwd = self.bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=recv_bwd)
                if j >= i:
                    self.bwd(i, bwd_j - i, None, True)
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)

            # insert recv_bwd into last fwd bwd
            recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=num_microbatches_remaining, pre_operation=fwd)

            for j in range(num_warmup_microbatches):
                bwd_j = j + num_microbatches_remaining
                recv_bwd = self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)
                bwd = self.bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=self.operations[i][RECVBWD][-2])
                self.bwd(i, bwd_j - i, None, True)
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)

            for j in range(i):
                bwd_j = j + num_warmup_microbatches + num_microbatches_remaining
                self.bwd(i, bwd_j - i, None, True)

        return self.operations


    def generate_n(self):
        total_num_microbatches = self.total_num_microbatches()
        for i in range(self.config.p):
            num_warmup_microbatches = self.num_warmup_microbatches(i)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            fwd = None
            bwd = None
            self.recv_fwd(pp_idx=i, batch_idx=0, start_time=0)
            for j in range(num_warmup_microbatches):
                fwd = self.fwd(i, j, self.operations[i][RECVFWD][-1])
                self.send_fwd(pp_idx=i, batch_idx=j, pre_operation=fwd)
                self.recv_fwd(pp_idx=i, batch_idx=j + 1, pre_operation=fwd)

            self.recv_bwd(pp_idx=i, batch_idx=0, pre_operation=fwd)
            for j in range(num_microbatches_remaining):
                fwd_j = j + num_warmup_microbatches
                bwd_j = j

                fwd = self.fwd(i, fwd_j, self.operations[i][RECVFWD][-1])
                self.send_fwd(pp_idx=i, batch_idx=fwd_j, pre_operation=fwd)
                self.recv_fwd(pp_idx=i, batch_idx=fwd_j + 1, pre_operation=fwd)

                bwd = self.bwd(i, bwd_j, self.operations[i][RECVBWD][-1])
                if j >= i:
                    self.bwd(i, bwd_j - i, None, True)
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)

            for j in range(num_warmup_microbatches):
                bwd_j = j + num_microbatches_remaining
                bwd = self.bwd(i, bwd_j, self.operations[i][RECVBWD][-1])
                self.bwd(i, bwd_j - i, None, True)
                self.send_bwd(pp_idx=i, batch_idx=bwd_j, pre_operation=bwd)
                self.recv_bwd(pp_idx=i, batch_idx=bwd_j + 1, pre_operation=bwd)

            for j in range(i):
                bwd_j = j + num_warmup_microbatches + num_microbatches_remaining
                self.bwd(i, bwd_j - i, None, True)

        return self.operations

    def bwd(self, pp_idx, batch_idx, pre_operation, use_x=False):
        vp_idx = self.get_chunk_idx(batch_idx, False)
        if not use_x:
            bwd = BWD(self.config, pp_idx, batch_idx, vp_idx, pre_operation=pre_operation)
        else:
            bwd = BWD_X(self.config, pp_idx, batch_idx, vp_idx, pre_operation=pre_operation)
        self.operations[pp_idx][COMPUTE].append(bwd)
        return bwd


class OperationExecutor:
    def __init__(self, config, operations) -> None:
        self.config = config
        self.operations = operations

    def compute_operations_count(self, i):
        return len(self.operations[i][COMPUTE])

    def execute(self):
        config = self.config
        operations = self.operations

        compute_idx_record = [0] * config.p
        score = 0

        while score < config.p:
            for i in range(config.p):
                j = compute_idx_record[i]
                if j == self.compute_operations_count(i):
                    continue
                while j < self.compute_operations_count(i):
                    op: Operation = operations[i][COMPUTE][j]
                    recv = op.pre_operation

                    recv_end_time = 0
                    if recv is not None:
                        pair_send = self.operations[recv.src_pp_idx][recv.pair_send_queue][recv.pair_send_idx]
                        if not pair_send.is_started():
                            break

                        recv_end_time = max(pair_send.start_time, recv.start_time) + pair_send.duration
                        pair_send.end_time = recv_end_time
                        recv.end_time = recv_end_time

                    op_start_time = recv_end_time
                    if j > 0:
                        pre_compute: Operation = operations[i][COMPUTE][j - 1]
                        op_start_time = max(recv_end_time, pre_compute.end_time)
                    op.start_time = op_start_time
                    op.update_end_time()
                    op.start_followed_ops()

                    j += 1
                compute_idx_record[i] = j
                if j == self.compute_operations_count(i):
                    score += 1
                # print(compute_idx_record)

    def export_trace(self, path):
        traces = []
        for _, v in self.operations.items():
            for item in v[COMPUTE]:
                traces.append(item.export_trace())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=4)

    def export_trace_full(self, path):
        traces = []
        for _, v in self.operations.items():
            for _, o in v.items():
                for item in o:
                    if item is None:
                        continue
                    traces.append(item.export_trace(False))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=4)

    def makespan(self):
        return self.operations[0][COMPUTE][-1].end_time
