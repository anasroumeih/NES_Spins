# Toric-code ground-manifold sampler

This upgrade replaces generic single-edge Metropolis updates for the toric code
with a physics-aware proposal kernel.

## Proposal mixture

For every chain and every Metropolis step, one NES replica is selected. The
proposal then is

- a **star flip** with probability `1 - toric_loop_prob - toric_single_flip_prob`;
- one of the two **non-contractible winding-loop flips** with total probability
  `toric_loop_prob`;
- an optional single-edge flip with probability `toric_single_flip_prob`.

For the intended fourfold ground-space calculation use:

```python
toric_loop_prob = 0.10
toric_single_flip_prob = 0.0
toric_cover_sectors = True
```

Star and loop flips preserve every plaquette eigenvalue `B_p`. Starting from a
flux-free configuration therefore restricts sampling to `B_p=+1`, where the
four toric-code ground states live. The loop flips connect the four topological
`Z2 x Z2` sectors, whereas star flips alone cannot.

The proposal is symmetric, so the Metropolis rule remains exactly

```text
accept = min(1, |det A(new)|^2 / |det A(old)|^2).
```

No overlap penalty, Gram--Schmidt step, or determinant jitter is introduced.

## Important scope

With `toric_single_flip_prob=0`, the sampler is deliberately restricted to the
flux-free manifold. This is appropriate for the first four ground states. It
is not appropriate for studying magnetic-flux/anyon excited states. To recover
an ergodic sampler over all plaquette sectors, set for example:

```python
toric_single_flip_prob = 0.05
toric_loop_prob = 0.05
```

The remaining 90% of proposals are still star flips.

## New diagnostics

The JSON history now contains:

```text
sampler_star_accept_rate
sampler_loop_accept_rate
sampler_single_flip_accept_rate
sampler_star_move_fraction
sampler_loop_move_fraction
sampler_single_flip_move_fraction
```

For a 4x4 run with `toric_loop_prob=0.10` and zero single flips, the move
fractions should be close to 0.90 star and 0.10 loop over a sufficiently long
run.
