# GTTM + Schrödinger Bridge: A Symbolic Music Generator

This repository is an implementation of the design specified in the "Software Design Specification: GTTM + Schrodinger Bridge" document. It outlines a modular software architecture for generating long-form, EDO-generic symbolic music in a functional style.

## Abstract

This project specifies a modular software architecture for generating long-form symbolic music (MIDI-oriented) using a hybrid of (i) GTTM-inspired structural energies, (ii) corpus-based statistical priors (optional, including neural predictive models), and (iii) Schrödinger bridge (SB) inference to produce coherent trajectories between endpoint passages. The design targets progressive rock / jazz fusion compositions of 5-15 minutes, supports multiple equal divisions of the octave (EDO) including 12-EDO and 19-EDO via a single parameter, and emphasizes functional programming principles. The immediate output is MIDI; audio rendering is a downstream concern.

## 1. Scope and Objectives

### 1.1 Primary Objectives
- Generate 5-15 minute pieces with multi-level structure (meter, grouping, harmonic motion, tension arcs) that remain novel, not merely derivative of a training corpus.
- Support two generation plans:
    - **Method A:** Generate start and end passages, then compute an SB "geodesic" between them under a combined prior.
    - **Method B:** Generate start and middle passages, compute SB from start to middle, then a second SB from middle back to start.
- Support both **Algorithmic** (GTTM + SB only) and **Hybrid** (with an optional neural predictive model) modes.
- Support EDO as a simple parameter `N` (not hard-coded to 12), with 12 and 19 as first-class use cases.

### 1.2 Non-goals (for the initial implementation)
- Direct audio generation.
- Full timbral modeling.

## 2. Design Principles

### 2.1 Functional Style and Modularity
The implementation should be as "functional-programming-ish" as is practical in Python:
- Prefer immutable configuration and data objects (`dataclasses` with `frozen=True`).
- Prefer pure functions (no hidden global state).
- Thread randomness explicitly (JAX-like).
- Define clear interfaces (protocol-style) for pluggable components.

### 2.2 Backend-Agnostic Numerics
Core algorithms should be written against a small backend interface, starting with NumPy and preserving the possibility of using JAX later.

## 3. Conceptual Pipeline

The system is organized as a layered pipeline. A typical run executes:

1.  `Configs + vocabularies + priors`
2.  `-> endpoint plan (A or B)`
3.  `-> build sparse layered graph of BeatState candidates`
4.  `-> solve Schrodinger bridge on that graph`
5.  `-> sample or MAP a BeatState trajectory`
6.  `-> decode BeatState trajectory to multi-track symbolic Score`
7.  `-> render Score to MIDI`

## 4. Core Representations

-   **EDO Configuration:** Pitch classes are in Z_N, and pitch heights are integers measured in EDO steps.
-   **Beat-level Structural State (`St`):** A compact, token-based representation of the musical state at a beat: `(meter_id, beat_in_bar, boundary_lvl, key_id, chord_id, role_id, head_id, groove_id)`.
-   **Score-level Representation (`NoteEvent`):** A tuple representing a single note: `(ton, toff, h, v, e, track)`.

## 13. Repository Organization

The implementation now lives under the `aimusic/` package and is grouped by responsibility:

| Package / Module                     | Responsibility                                               |
| ------------------------------------ | ------------------------------------------------------------ |
| `aimusic.core.config`                | Immutable configuration dataclasses.                         |
| `aimusic.core.core_types`            | Canonical shared types such as `BeatState`, `Layer`, `Edge`, and `Score`. |
| `aimusic.core.vocab`                 | Token vocabularies for meters, grooves, chords, keys, and roles. |
| `aimusic.core.rng`                   | Deterministic RNG helpers for pure sampling code.            |
| `aimusic.theory.edo`                 | EDO pitch math and rendering helpers.                        |
| `aimusic.theory.tonal`               | Tonal system definition, chord templates, tonal distances.   |
| `aimusic.scoring.gttm_features`      | Feature functions and weighted energy computation.           |
| `aimusic.scoring.priors`             | `NullPrior`, placeholder `NeuralPrior`, manifests, and prior scoring helpers. |
| `aimusic.scoring.rhythm_features`    | Compatibility facade over the GTTM beat-state scoring layer. |
| `aimusic.planning.candidates`        | Hard gating and candidate proposal functions.                |
| `aimusic.planning.graph`             | Layer expansion, sparse edge building, pruning.              |
| `aimusic.planning.plans`             | Endpoint generation, section metadata, and Method A orchestration. |
| `aimusic.planning.sb`                | Schrödinger bridge solver, bridge extraction, sampling, and MAP path extraction. |
| `aimusic.app.main`                   | Application / demo entrypoints.                              |

The test suite lives under `tests/` and imports the packaged modules directly.

## 17. Implementation Checklist (Practical)

A recommended implementation order:

1.  Define configs and token vocabularies (12-EDO first, then 19-EDO).
2.  Implement GTTM feature energies and tonal distance metric (simple, rule-based).
3.  Implement candidate generation and sparse graph builder with pruning.
4.  Implement SB solver on sparse edges (NumPy backend).
5.  Implement Method A plan, then Method B.
6.  Implement decoder (drums, bass, comping, lead) and MIDI rendering.
7.  Add the placeholder `NeuralPrior` seam and artifact contract.
8.  Integrate the external neural prior implementation when it is available.
9.  Add section-wise SB and richer diagnostics.
