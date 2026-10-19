# CuS–ZnS 2D percolation prototype

This is a deliberately simple research prototype for 2D CuS grain morphology,
contact-network connectivity and a four-point-probe electrical calculation.

## Install

```bash
python -m pip install numpy scipy networkx matplotlib
```

## Run the demo

```bash
python demo.py
```

The demo uses a small domain and reduced probe spacing so it runs quickly.
Its output is **not a calibrated prediction** of experimental resistance.

## Current assumptions and limitations

- 2D circles represent CuS-rich grains in a ZnS-rich background.
- The diameter distribution is currently uniform over the chosen interval;
  replace it with an SEM-measured distribution when available.
- `allow_overlap=True` is used in the demo. In this case the nominal area
  fraction counts overlapping circle areas multiple times and is not the
  true occupied area fraction.
- `allow_overlap=False` uses random sequential addition and will jam before
  high area fractions are reached.
- Contacts are geometrical only; there is no tunnelling.
- The current electrical model treats each grain as equipotential and places
  a grain-boundary resistor between contacting grains. It does **not yet**
  separately resolve intragrain resistance. A physically consistent model
  with both grain and boundary resistance requires a multi-terminal grain
  model or internal resistive elements; simply adding a full grain resistance
  to every contact edge would double-count grain contributions.
- Four ideal collinear point probes are approximated as nodes coupled to
  grains intersecting their contact radius. A real four-point probe also
  requires calibrated probe contact geometry and finite-sample corrections.
- The domain dimensions and probe spacing must use the same units.

## Recommended next steps

1. Validate the graph using hand-built test morphologies.
2. Validate the resistor solver on simple series/parallel networks.
3. Add a physically justified intragrain resistance representation.
4. Estimate grain-size distribution from SEM.
5. Test finite-size convergence and compare simulated sheet resistance with
   experimental measurements.


## Electrical solver validation

The resistor-network solver can be checked against known circuits before using
it to interpret film simulations. From this directory, run:

```bash
PYTHONPATH=. pytest -q
```

The analytical tests cover a single resistor, a two-resistor series circuit,
two equal branches in parallel, a disconnected source/sink pair, and rejection
of non-positive resistance values. The solver returns `connected=False` for a
disconnected source/sink pair instead of reporting a finite resistance. These
tests validate the numerical network solver; they do not yet validate the
physical assumptions of the CuS grain/contact model or calibrate it to a real
four-point-probe instrument.
