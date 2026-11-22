# Linux CCTM auxiliary parity adjustment

The historical Windows CCTM checkpoint stored the complete wealth path,
running maximum, crossing indicators, atom counts and threshold, but its
optional worst_parameter array was entirely NaN.

Linux recomputation reproduced every scientific detection quantity exactly:
the maximum absolute wealth-path difference was 0, alarm times were identical,
and atom counts were identical. The new implementation additionally produced
the previously unavailable worst-parameter diagnostic.

Replay parity therefore normalizes only the all-NaN legacy auxiliary field
before computing the semantic hash. No detector, threshold, wealth path,
alarm, seed, raw trajectory or formal result was modified.
