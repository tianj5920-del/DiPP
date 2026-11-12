from pathlib import Path
import json
from dipp_label.workflow import run


def test_end_to_end_nested_pipeline(config,tmp_path):
    data=Path(__file__).resolve().parents[1]/'dataset/example'
    output=tmp_path/'run'
    rows=run(data,output,config,['dipp_label','lr','random'],validate=True,explain=False)
    assert len(rows)==6
    assert {r['phase'] for r in rows}=={'test','cv'}
    assert json.loads((output/'completion.json').read_text())['status']=='completed'
    assert json.loads((output/'run_manifest.json').read_text())['synthetic'] is True
    assert all('HC_auc_ovr' in r['mean'] and 'macro_auc_ovo' in r['mean'] for r in rows)
    assert not list(output.glob('*.npy'))
