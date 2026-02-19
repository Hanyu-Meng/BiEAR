import h5py
import numpy as np

# 你原来从 data_save import * 里来的函数
# 这里假设 load_dataset_from_dir(dataset_dir, max_samples=...) 会返回:
#   (x1, x2, y) 或 (x1, x2, x3, y)
from data_save import load_dataset_from_dir


def _auto_chunks(arr, chunk_rows: int):
    arr = np.asarray(arr)
    return (min(chunk_rows, arr.shape[0]),) + tuple(arr.shape[1:])


def save_dataset_to_h5(
    dataset_dir=None,
    h5_path=None,
    max_samples=None,
    compression=None,

    # 兼容“直接给数组”的新用法
    out_h5=None,
    x1=None, x2=None, y=None, x3=None,

    chunk_rows=256,
    comp_level=4,
):
    """
    兼容两种用法：
    A) save_dataset_to_h5(dataset_dir=..., h5_path=..., ...)
       -> 从目录加载，再写H5
    B) save_dataset_to_h5(out_h5=..., x1=..., x2=..., y=..., ...)
       -> 直接把数组写进H5
    """

    # ---------- 1) 解析输出路径 ----------
    if out_h5 is None:
        out_h5 = h5_path
    if out_h5 is None:
        raise ValueError("Please provide `out_h5` or `h5_path`.")

    # ---------- 2) 如果没给数组，则从目录加载 ----------
    if x1 is None or x2 is None or y is None:
        if dataset_dir is None:
            raise ValueError("You didn't provide x1/x2/y, so `dataset_dir` must be provided.")

        print(f"[save_dataset_to_h5] Loading from dir: {dataset_dir}")
        loaded = load_dataset_from_dir(dataset_dir, max_samples=max_samples)

        # 兼容 load_dataset_from_dir 返回 (x1,x2,y) 或 (x1,x2,x3,y)
        if len(loaded) == 3:
            x1, x2, y = loaded
            x3 = None
        elif len(loaded) == 4:
            x1, x2, x3, y = loaded
        else:
            raise ValueError(f"Unexpected return from load_dataset_from_dir: len={len(loaded)}")

    # ---------- 3) 写H5 ----------
    x1 = np.asarray(x1, dtype=np.float32)
    x2 = np.asarray(x2, dtype=np.float32)
    y  = np.asarray(y,  dtype=np.float32)
    if x3 is not None:
        x3 = np.asarray(x3, dtype=np.float32)

    N = x1.shape[0]
    chunk_rows = min(int(chunk_rows), int(N))

    print(f"[save_dataset_to_h5] Saving -> {out_h5}")
    print("  x1:", x1.shape, "x2:", x2.shape, "y:", y.shape, "x3:", (None if x3 is None else x3.shape))

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("x1", data=x1, chunks=_auto_chunks(x1, chunk_rows),
                         compression=compression, compression_opts=(comp_level if compression else None))
        f.create_dataset("x2", data=x2, chunks=_auto_chunks(x2, chunk_rows),
                         compression=compression, compression_opts=(comp_level if compression else None))
        if x3 is not None:
            f.create_dataset("x3", data=x3, chunks=_auto_chunks(x3, chunk_rows),
                             compression=compression, compression_opts=(comp_level if compression else None))
        f.create_dataset("y",  data=y,  chunks=_auto_chunks(y,  chunk_rows),
                         compression=compression, compression_opts=(comp_level if compression else None))

    print("[save_dataset_to_h5] Done.")