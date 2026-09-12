"""Behavioral tests for the meta/ranks/<idx>/workers/<widx>/pid FUSE dynamic subtree.

Exercises `TrainctlFS` directly against a `TrainctlRuntime` with hand-published
rank state, rather than mounting real FUSE -- the routing logic under test lives
entirely in `virtual_fs.generate`/`is_virtual_dir`/`TrainctlFS.readdir`, not in the
kernel mount.
"""

from trainctl.config import TrainctlConfig
from trainctl.fuse.filesystem import TrainctlFS
from trainctl.lightning_backend import load_backend
from trainctl.runtime import virtual_fs
from trainctl.runtime.runtime import TrainctlRuntime
from trainctl.runtime.state import RankInfo


def _fs_with_rank(worker_pids: tuple[int, ...] = (111, 222)) -> TrainctlFS:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    rank = RankInfo(
        global_rank=0,
        local_rank=0,
        pid=999,
        hostname="h",
        device="cpu",
        worker_pids=worker_pids,
    )
    runtime.state.update(world_size=1, ranks=(rank,))
    return TrainctlFS(runtime)


def test_workers_dir_listed_under_rank() -> None:
    fs = _fs_with_rank()
    assert fs.readdir("/meta/ranks/000", 0) == [
        ".",
        "..",
        "pid",
        "hostname",
        "global-rank",
        "local-rank",
        "device",
        "workers",
    ]


def test_worker_indices_listed() -> None:
    fs = _fs_with_rank((111, 222))
    assert fs.readdir("/meta/ranks/000/workers", 0) == [".", "..", "000", "001"]


def test_worker_indices_empty_when_no_workers() -> None:
    fs = _fs_with_rank(())
    assert fs.readdir("/meta/ranks/000/workers", 0) == [".", ".."]


def test_worker_fields_listed_under_worker_index() -> None:
    fs = _fs_with_rank((111,))
    assert fs.readdir("/meta/ranks/000/workers/000", 0) == [".", "..", "pid"]


def test_worker_pid_file_content() -> None:
    fs = _fs_with_rank((111, 222))
    assert (
        virtual_fs.generate(fs._runtime, "/meta/ranks/000/workers/000/pid") == b"111\n"
    )
    assert (
        virtual_fs.generate(fs._runtime, "/meta/ranks/000/workers/001/pid") == b"222\n"
    )


def test_worker_pid_file_read_and_getattr_round_trip() -> None:
    fs = _fs_with_rank((111,))
    attrs = fs.getattr("/meta/ranks/000/workers/000/pid")
    assert attrs["st_size"] == len(b"111\n")

    fh = fs.open("/meta/ranks/000/workers/000/pid", 0)
    try:
        assert fs.read("/meta/ranks/000/workers/000/pid", 4096, 0, fh) == b"111\n"
    finally:
        fs.release("/meta/ranks/000/workers/000/pid", fh)


def test_worker_index_out_of_range_is_not_found() -> None:
    fs = _fs_with_rank((111,))
    assert virtual_fs.generate(fs._runtime, "/meta/ranks/000/workers/001/pid") is None
    assert not virtual_fs.is_virtual_dir(fs._runtime, "/meta/ranks/000/workers/001")


def test_rank_index_out_of_range_is_not_found() -> None:
    fs = _fs_with_rank((111,))
    assert virtual_fs.generate(fs._runtime, "/meta/ranks/007/workers/000/pid") is None
    assert not virtual_fs.is_virtual_dir(fs._runtime, "/meta/ranks/007/workers")
