"""Create fictitious PPMI-style tables using only seeded random numbers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from . import CLASSES


def generate_example(folder, seed=20261112, overwrite=False):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not overwrite:
        raise ValueError("Example directory is not empty; use --overwrite explicitly")
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count, width = 216, 64
    labels = np.repeat([0, 1, 2], [48, 96, 72])
    labels = labels[rng.permutation(count)]
    ids = [f"SYN_{i+1:04d}" for i in range(count)]
    proteins = [f"SYN_OID{i+1:04d}" for i in range(width)]
    people = []
    records = []
    means = np.r_[np.full(24, 1.5), np.full(16, .2), np.full(24, -.55)]
    loadings = rng.normal(0, .15, (4, width))
    x = 5 + means + rng.normal(size=(count, 4)) @ loadings + rng.normal(0, .8, (count, width))
    for label in range(3):
        x[labels == label, label*4:(label+1)*4] += 1.1
        x[labels == label, 24+label*3:27+label*3] += .7
        x[labels == label, 40+label*3:43+label*3] += .35
    lod = 5 + rng.normal(0, .06, (count, width))
    missing = rng.random((count, width)) < .015
    qc = rng.random((count, width)) > .005
    x[missing] = np.nan
    def add(person, project, npx, limit, passed, event="BL"):
        for j in range(width):
            records.append(dict(PATNO=person, PROJECT=project, EVENT_ID=event, BIOFLUID="Plasma", SampleType="SAMPLE",
                AssayType="assay", OlinkID=proteins[j], NPX=npx[j], LOD=limit[j], QC_Pass=int(passed[j])))
    for i, person in enumerate(ids):
        people.append(dict(PATNO=person, ENROLL_CAT=CLASSES[labels[i]], SEX="M" if rng.random()<.5 else "F",
                           DM_CORE_EVIDENCE=int(i%4 != 0), GLUCOSE_MMOL_L=8.0 if i%4 == 0 else 5.5))
        project = "314" if rng.random()<.55 else "293"
        add(person, project, x[i], lod[i], qc[i])
        if i < 8:
            other = "293" if project == "314" else "314"
            # Two synthetic projects for the same person exercise priority handling.
            add(person, other, x[i]+.03, lod[i]+.03, qc[i])
    for i in range(3):
        person = f"SYN_EXCLUDED_{i+1:02d}"
        people.append(dict(PATNO=person, ENROLL_CAT=CLASSES[i], SEX="F", DM_CORE_EVIDENCE=int(i!=2), GLUCOSE_MMOL_L=5.0))
        add(person, "314", 5+means, np.full(width,5.), np.full(width,i!=1), "M12" if i==0 else "BL")
    pd.DataFrame(people).to_csv(folder/'participants.csv',index=False,lineterminator='\n')
    pd.DataFrame(dict(OlinkID=proteins, Assay=[f"SyntheticProtein{i+1:03d}" for i in range(width)])).to_csv(folder/'assays.csv',index=False,lineterminator='\n')
    pd.DataFrame(records).to_csv(folder/'plasma.csv',index=False,float_format='%.8f',lineterminator='\n')
    manifest=dict(synthetic=True, generator='dipp_label.synthetic.generate_example', seed=seed,
        expected_eligible_participants=count, assays=width, class_order=list(CLASSES),
        purpose='Schema and execution example only; all people, proteins and measurements are invented.',
        source='Seeded random numbers and explicit toy class shifts; no real participant records or fitted statistics used.',
        sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('participants.csv','assays.csv','plasma.csv')})
    (folder/'SYNTHETIC_DATA.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
