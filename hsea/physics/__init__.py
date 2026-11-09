"""FBP-inspired cell fire spread (T0 semantics).

Each cell has fuel, moisture, and burn state. Burning cells expose their
eight neighbors every step; exposure accumulates at a rate-of-spread set
by fuel x dryness x wind alignment, and the cell ignites past a threshold.
Fully deterministic: no RNG, only arithmetic. Higher tiers render this
model; they never redefine it.
"""