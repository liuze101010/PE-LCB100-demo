# Multi-polymer short-compute validation

Validation date: 11 August 2026

This check exercises the real preparation path rather than the read-only
PE-LCB replay. Each passing case was built by EMC, converted from EMC
PSF/PDB/parameter outputs to GROMACS files, accepted by `gmx grompp`, and run
for 50 steepest-descent steps. These small systems test software integration
and parameter availability. They do not establish equilibrated material
properties.

## Runtime and protocol

- EMC 9.4.4, official executable built 21 July 2026
- OPLS-AA/2024 selected through EMC
- GROMACS 2020.6-MODIFIED for the local short run
- 16 short chains for passing cases
- initial density: 0.55 g/cm3
- minimization tolerance: 1000 kJ mol-1 nm-1
- 50 steepest-descent steps

## Completed short calculations

| Polymer | Repeat-unit input | Box edge (nm) | Final potential energy (kJ/mol) | Maximum force (kJ mol-1 nm-1) | Result |
| --- | --- | ---: | ---: | ---: | --- |
| PE | `*CC*` | 2.12411 | 1.4502462e+03 | 8.8776086e+02 | Finished |
| PP | `*CC(C)*` | 2.60682 | 5.1812769e+03 | 8.6893481e+02 | Finished |
| PTFE | `*C(F)(F)C(F)(F)*` | 2.74916 | 1.7742568e+03 | 9.3686853e+02 | Finished |
| PS | `*CC(c1ccccc1)*` | 3.95420 | 1.7726473e+04 | 8.8497913e+02 | Finished |
| PMMA | `*CC(C)(C(=O)OC)*` | 3.42402 | 5.5843105e+03 | 7.8046228e+02 | Finished |

Every row above has a GROMACS log containing `Finished mdrun`. The maximum
force in every case is below the configured minimization tolerance.

## Correctly blocked at parameterization

| Polymer | Repeat-unit input | Blocking evidence from EMC | Result |
| --- | --- | --- | --- |
| PEO | `*CCO*` | missing increments `{c4, o2e}` and `{c4, h1co}` | Review required |
| PVC | `*CC(Cl)*` | missing increment `{c4, cl1}` | Review required |
| POM | `*CO*` | missing increment `{c4, h1co}` | Review required |
| PA6 | `*NCCCCCC(=O)*` | missing increments `{n3h, c4}` and `{n3h, c3oam}` | Review required |
| PET | `*OCCOC(=O)c1ccc(C(=O)*)cc1` | missing increments `{o2e, c4}` and `{c4, o2s}` | Review required |
| PPS | `*Sc1ccc(*)cc1` | no OPLS rule for two repeat-unit sites | Review required |
| PCL | `*OCCCCCC(=O)*` | missing increment `{o2e, c4}` | Review required |

No blocked case entered GROMACS. The current system therefore supports the
listed completed cases as integration-tested routes and treats the remaining
materials as parameterization work, not successful simulations.

## Defects found and fixed

1. The production worker resolved `POLYMER_AGENT_EMC_ROOT` but did not export
   EMC's required `EMC_ROOT` variable to child processes.
2. EMC can report a parameterization abort while returning a successful process
   exit code. The worker now requires `.psf.gz`, `.pdb.gz`, and `.params`
   artifacts before bridge generation.
3. EMC's Windows Perl setup cannot reliably resolve a runtime root containing
   whitespace. The worker now stops with a direct installation-path message.
4. The managed installer referenced an expired 2025 executable. It now uses
   the official July 2026 x86_64 and aarch64 Linux archives with fixed SHA-256
   values and records skipped external symbolic links.

The repository regression suite passed 92/92 after these changes. Dependency
checking and the repository security scan also passed.
