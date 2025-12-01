"""PK/PD simulator for a T cell engager.

The script implements a light-weight deterministic PK/PD model that can be
explored from the command line without external dependencies. It uses a
one-compartment PK model with first-order elimination and a PD module that
captures tumor growth and T cell stimulation driven by drug exposure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PKParameters:
    """Pharmacokinetic parameters for the T cell engager."""

    clearance: float
    volume: float


@dataclass
class PDParameters:
    """Pharmacodynamic parameters governing tumor and effector dynamics."""

    tumor_growth_rate: float
    tumor_carrying_capacity: float
    kmax: float
    ec50: float
    hill: float
    t_cell_homeostasis: float
    t_cell_stimulation: float
    t_cell_carrying_capacity: float


@dataclass
class SimulationState:
    time: float
    amount: float
    tumor_cells: float
    effector_cells: float


@dataclass
class SimulationResult:
    times: List[float]
    concentrations: List[float]
    tumor_cells: List[float]
    effector_cells: List[float]


class PKPDSimulator:
    """Simple explicit-Euler simulator for PK/PD exploration."""

    def __init__(
        self,
        pk: PKParameters,
        pd: PDParameters,
        dose_amount: float,
        dose_interval: float,
        n_doses: int,
        initial_tumor: float,
        initial_effectors: float,
        dt: float,
    ) -> None:
        self.pk = pk
        self.pd = pd
        self.dose_amount = dose_amount
        self.dose_interval = dose_interval
        self.n_doses = n_doses
        self.initial_tumor = initial_tumor
        self.initial_effectors = initial_effectors
        self.dt = dt
        self.dosing_events = self._build_dosing_schedule()

    def _build_dosing_schedule(self) -> Dict[float, float]:
        events: Dict[float, float] = {}
        for i in range(self.n_doses):
            t = round(i * self.dose_interval, 10)
            events[t] = events.get(t, 0.0) + self.dose_amount
        return events

    def _add_dose(self, time: float, amount: float) -> float:
        if time in self.dosing_events:
            amount += self.dosing_events[time]
        return amount

    def _pk_derivative(self, amount: float) -> float:
        ke = self.pk.clearance / self.pk.volume
        return -ke * amount

    def _effect(self, concentration: float) -> float:
        if concentration <= 0:
            return 0.0
        c_hill = concentration ** self.pd.hill
        ec50_hill = self.pd.ec50 ** self.pd.hill
        return c_hill / (ec50_hill + c_hill)

    def _pd_derivatives(
        self, concentration: float, tumor: float, effectors: float
    ) -> Tuple[float, float]:
        effect = self._effect(concentration)
        tumor_growth = self.pd.tumor_growth_rate * tumor * (
            1.0 - tumor / self.pd.tumor_carrying_capacity
        )
        tumor_kill = self.pd.kmax * effect * tumor
        d_tumor = tumor_growth - tumor_kill

        homeostasis = self.pd.t_cell_homeostasis * effectors * (
            1.0 - effectors / self.pd.t_cell_carrying_capacity
        )
        stimulation = self.pd.t_cell_stimulation * effect * effectors
        d_effectors = homeostasis + stimulation
        return d_tumor, d_effectors

    def run(self, duration: float) -> SimulationResult:
        state = SimulationState(
            time=0.0,
            amount=0.0,
            tumor_cells=self.initial_tumor,
            effector_cells=self.initial_effectors,
        )

        times: List[float] = []
        concentrations: List[float] = []
        tumors: List[float] = []
        effectors: List[float] = []

        n_steps = int(duration / self.dt)
        for step in range(n_steps + 1):
            current_time = round(step * self.dt, 10)
            state.amount = self._add_dose(current_time, state.amount)
            concentration = state.amount / self.pk.volume
            d_amount = self._pk_derivative(state.amount)
            d_tumor, d_effectors = self._pd_derivatives(
                concentration, state.tumor_cells, state.effector_cells
            )

            times.append(current_time)
            concentrations.append(concentration)
            tumors.append(state.tumor_cells)
            effectors.append(state.effector_cells)

            state.amount = max(state.amount + d_amount * self.dt, 0.0)
            state.tumor_cells = max(state.tumor_cells + d_tumor * self.dt, 0.0)
            state.effector_cells = max(
                state.effector_cells + d_effectors * self.dt, 0.0
            )

        return SimulationResult(times, concentrations, tumors, effectors)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore PK/PD for a T cell engager using a one-compartment model "
            "and Hill-based cytotoxic effect."
        )
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation days")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step (days)")
    parser.add_argument("--dose", type=float, default=50.0, help="Dose amount (mg)")
    parser.add_argument("--interval", type=float, default=7.0, help="Dosing interval (days)")
    parser.add_argument("--n-doses", type=int, default=4, help="Number of doses")
    parser.add_argument("--cl", type=float, default=0.8, help="Clearance (L/day)")
    parser.add_argument("--v", type=float, default=5.0, help="Volume of distribution (L)")
    parser.add_argument(
        "--tumor",
        type=float,
        default=1.0,
        help="Initial tumor burden (normalized units)",
    )
    parser.add_argument(
        "--effectors",
        type=float,
        default=0.5,
        help="Initial effector T cell pool (normalized units)",
    )
    parser.add_argument(
        "--ec50",
        type=float,
        default=0.5,
        help="EC50 driving tumor kill and T cell stimulation (mg/L)",
    )
    parser.add_argument("--kmax", type=float, default=0.7, help="Max kill rate (1/day)")
    parser.add_argument("--hill", type=float, default=1.2, help="Hill coefficient")
    parser.add_argument(
        "--growth-rate",
        type=float,
        default=0.15,
        help="Tumor logistic growth rate (1/day)",
    )
    parser.add_argument(
        "--tumor-capacity",
        type=float,
        default=10.0,
        help="Tumor carrying capacity (normalized units)",
    )
    parser.add_argument(
        "--t-homeostasis",
        type=float,
        default=0.05,
        help="Effector T cell homeostatic growth (1/day)",
    )
    parser.add_argument(
        "--t-stim",
        type=float,
        default=0.2,
        help="Drug-driven effector expansion (1/day)",
    )
    parser.add_argument(
        "--t-capacity",
        type=float,
        default=2.0,
        help="Effector T cell carrying capacity (normalized units)",
    )
    return parser.parse_args()


def _summarize(result: SimulationResult) -> str:
    max_conc = max(result.concentrations)
    final_tumor = result.tumor_cells[-1]
    final_effectors = result.effector_cells[-1]
    response = (
        "Tumor eradication" if final_tumor < 0.01 else "Tumor persists"
    )
    return (
        f"Max concentration: {max_conc:.3f} mg/L\n"
        f"Final tumor: {final_tumor:.3f} (normalized)\n"
        f"Final effector T cells: {final_effectors:.3f} (normalized)\n"
        f"Qualitative response: {response}"
    )


def main() -> None:
    args = _parse_args()
    pk_params = PKParameters(clearance=args.cl, volume=args.v)
    pd_params = PDParameters(
        tumor_growth_rate=args.growth_rate,
        tumor_carrying_capacity=args.tumor_capacity,
        kmax=args.kmax,
        ec50=args.ec50,
        hill=args.hill,
        t_cell_homeostasis=args.t_homeostasis,
        t_cell_stimulation=args.t_stim,
        t_cell_carrying_capacity=args.t_capacity,
    )

    simulator = PKPDSimulator(
        pk=pk_params,
        pd=pd_params,
        dose_amount=args.dose,
        dose_interval=args.interval,
        n_doses=args.n_doses,
        initial_tumor=args.tumor,
        initial_effectors=args.effectors,
        dt=args.dt,
    )

    result = simulator.run(duration=args.duration)
    print(_summarize(result))


if __name__ == "__main__":
    main()
