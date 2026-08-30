# Memory optimization notes

Global memory is backed by device DRAM and is most efficient when neighboring lanes access neighboring addresses. A row-major copy naturally coalesces a row. A naive transpose reads rows coalesced but writes a column-strided pattern, so its write transactions are less efficient.

The tiled transpose first loads a 32 by 32 tile into shared memory, synchronizes, and writes the transposed tile. Shared memory changes the global-memory access pattern, but a square shared tile can create bank conflicts when a warp reads a column. The padded form declares `tile[32][33]`; the extra column changes the bank mapping so successive elements of a transposed access do not repeatedly target the same bank.

Padding is not a promise of a speedup. It adds a small amount of shared storage and its value depends on architecture, compiler, shape, and occupancy. The repository therefore keeps both tiled forms and reports their measured statistics rather than claiming a universal ranking.

