# CUDA execution model

CUDA launches a grid of thread blocks. A block is scheduled as a unit on one streaming multiprocessor, while threads inside the block execute in warps of 32 lanes. A thread computes an index from its block and lane coordinates; a grid-stride loop or a bounds check handles inputs that are not exact multiples of the launch shape.

Warps execute the same instruction stream under SIMT. Divergent branches serialize the active lane groups, so the kernels keep boundary checks short and place the regular work in the main path. The reduction implementation uses a warp-level shuffle to exchange registers without a shared-memory round trip.

The launch configuration is part of the experiment. A 256-thread one-dimensional block is used for vector add and reduction, while transpose uses a 32 by 32 tile and GEMM uses a 16 by 16 tile. These are educational baselines, not universal optimums; the benchmark should be rerun when the GPU, compiler, or shape changes.

