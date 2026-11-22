# Resolution of six numerical UNKNOWN records

The six method records arose from three healthy trajectories.  Both full-domain
engines stopped at the same near-collision because binary64 could not certify
the order of a new rank root relative to an old atom.

For each affected method and every monitoring time,
`inf_a M_t(a) <= M_t(0.9)`.  The witness ranks were computed by exact comparison
of the stored binary64 inputs as rational numbers.  Wealth was recomputed at
100 and 160 decimal digits and the resulting float64 paths agreed within the
recorded tolerance.  Every witness path remained at least one full wealth unit
below its alarm threshold; the observed margins were much larger.

Accordingly these records are certified only as non-alarms.  The stored path is
a witness upper bound and MUST NOT be described as the recovered full profile.
Original UNKNOWN traces and rows remain under `original_unknown_records/`.
