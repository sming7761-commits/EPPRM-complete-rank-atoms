# Worst-parameter representation caveat

Each full rank atom may be a singleton at an algebraic residual-equality root.
The profiler updates wealth using the mathematical tie-rank at this root,
but the binary64 coordinate saved in `worst_parameter` is a rounded
representation of that root. Recomputing floating-point residual ranks at
this rounded coordinate can select a different adjacent rank and wealth.
In eight pipeline smoke trajectories, sampled worst-parameter replay
differences up to 0.00462 were observed despite the minimum-wealth curves
being stable under the implemented float event processing.

Therefore worst-parameter coordinates in the smoke results are diagnostic
approximations and must not be used as formal rational certificates. The
production engines are explicitly tagged `GENERAL_POSITION_FLOAT_*`, not
`NUMERICALLY_CERTIFIED`. Full algebraic-root location or certified interval
replay would be required to make each recorded singleton parameter exact.

The EP-PRM minimizer recorder was corrected to select the final post-merge
minimum state rather than an intermediate representative. All eight smoke
EP-PRM full paths were recomputed with unchanged path wealth values (max
absolute difference 0), with the revised diagnostic parameter coordinate.
