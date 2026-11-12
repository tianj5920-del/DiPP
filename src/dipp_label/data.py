"""Explicit CSV input contract; no implicit local paths or remote data access."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import CLASSES


@dataclass
class Cohort:
    ids: np.ndarray
    y: np.ndarray
    project: np.ndarray
    sex: np.ndarray
    npx: np.ndarray
    lod: np.ndarray
    assay_ids: np.ndarray
    assay_names: np.ndarray
    synthetic: bool = False

    def take(self, indices):
        return Cohort(self.ids[indices], self.y[indices], self.project[indices], self.sex[indices],
                      self.npx[indices], self.lod[indices], self.assay_ids, self.assay_names, self.synthetic)

    def __len__(self):
        return len(self.y)


def _required(frame, names, filename):
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename}: missing columns {missing}")


def load_cohort(folder, minimum_sample_coverage=0.8):
    """Select baseline plasma, mask QC failures, then prefer project 314 over 293.

    DM_CORE_EVIDENCE is a participant-level flag prepared from medication and
    indication records; GLUCOSE_MMOL_L is a valid observed glucose maximum.
    These are eligibility inputs and are never passed to the classifier.
    """
    folder = Path(folder)
    people = pd.read_csv(folder / "participants.csv", dtype={"PATNO": str, "ENROLL_CAT": str, "SEX": str})
    assays = pd.read_csv(folder / "assays.csv", dtype=str)
    plasma = pd.read_csv(folder / "plasma.csv", dtype={"PATNO": str, "OlinkID": str, "PROJECT": str})
    _required(people, ["PATNO", "ENROLL_CAT", "SEX", "DM_CORE_EVIDENCE", "GLUCOSE_MMOL_L"], "participants.csv")
    _required(assays, ["OlinkID", "Assay"], "assays.csv")
    _required(plasma, ["PATNO", "PROJECT", "EVENT_ID", "BIOFLUID", "SampleType", "AssayType", "OlinkID", "NPX", "LOD", "QC_Pass"], "plasma.csv")
    if people.PATNO.isna().any() or people.PATNO.duplicated().any() or assays.OlinkID.isna().any() or assays.OlinkID.duplicated().any():
        raise ValueError("Participant and assay metadata keys must be unique and nonmissing")
    people.ENROLL_CAT = people.ENROLL_CAT.replace({"NC": "HC", "Prodromal PD": "Prodromal"})
    if not people.DM_CORE_EVIDENCE.isin([0, 1]).all() or not plasma.QC_Pass.isin([0, 1]).all():
        raise ValueError("DM_CORE_EVIDENCE and QC_Pass must be explicit 0/1 flags")
    mask = (plasma.PROJECT.isin(["314", "293"]) & plasma.EVENT_ID.eq("BL") &
            plasma.BIOFLUID.str.casefold().eq("plasma") & plasma.SampleType.eq("SAMPLE") & plasma.AssayType.eq("assay"))
    plasma = plasma.loc[mask].copy()
    if not plasma.PATNO.isin(people.PATNO).all() or not plasma.OlinkID.isin(assays.OlinkID).all():
        raise ValueError("Plasma contains participant/assay keys absent from metadata")
    if plasma.duplicated(["PATNO", "PROJECT", "OlinkID"]).any():
        raise ValueError("Resolve technical replicates before import: duplicate participant/project/assay")
    for column in ("NPX", "LOD"):
        plasma[column] = pd.to_numeric(plasma[column], errors="raise").replace([np.inf, -np.inf], np.nan)
        plasma.loc[plasma.QC_Pass.eq(0), column] = np.nan
    # A common measured assay catalog, independent of participant outcomes.
    project_assays = [set(plasma.loc[plasma.PROJECT.eq(s), "OlinkID"]) for s in ("314", "293")]
    active = [s for s in project_assays if s]
    if not active:
        raise ValueError("No baseline plasma from projects 314/293")
    shared = set.intersection(*active)
    catalog = assays[assays.OlinkID.isin(shared)].sort_values("OlinkID")
    if catalog.empty:
        raise ValueError("No shared assay catalog")
    npx = plasma.pivot(index=["PATNO", "PROJECT"], columns="OlinkID", values="NPX").reindex(columns=catalog.OlinkID)
    lod = plasma.pivot(index=["PATNO", "PROJECT"], columns="OlinkID", values="LOD").reindex(columns=catalog.OlinkID)
    passing = npx.notna().mean(axis=1).ge(minimum_sample_coverage)
    people = people[people.ENROLL_CAT.isin(CLASSES) & (people.DM_CORE_EVIDENCE.eq(1) |
                    pd.to_numeric(people.GLUCOSE_MMOL_L, errors="raise").ge(7.0))].sort_values("PATNO")
    selected = []
    for row in people.itertuples():
        for project in ("314", "293"):
            key = (row.PATNO, project)
            if key in passing.index and bool(passing.loc[key]):
                selected.append((row, key))
                break
    if not selected:
        raise ValueError("No participants pass eligibility and plasma coverage")
    keys = [key for _, key in selected]
    marker = folder / "SYNTHETIC_DATA.json"
    synthetic = marker.is_file() and json.loads(marker.read_text()).get("synthetic") is True
    return Cohort(np.array([r.PATNO for r, _ in selected]), np.array([CLASSES.index(r.ENROLL_CAT) for r, _ in selected]),
                  np.array([k[1] for k in keys]), np.array([r.SEX for r, _ in selected]),
                  npx.loc[keys].to_numpy(float), lod.loc[keys].to_numpy(float),
                  catalog.OlinkID.to_numpy(), catalog.Assay.to_numpy(), synthetic)
