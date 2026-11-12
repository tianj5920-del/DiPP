import argparse
import json
from pathlib import Path
from .config import Config
from .data import load_cohort
from .synthetic import generate_example


def main():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    except ImportError:
        pass
    parser=argparse.ArgumentParser(description='DiPP-Label: data preparation, training, evaluation and protein attribution')
    commands=parser.add_subparsers(dest='command',required=True)
    example=commands.add_parser('make-example',help='Generate wholly synthetic PPMI-style CSVs')
    example.add_argument('--out',default='dataset/example');example.add_argument('--seed',type=int,default=20261112)
    example.add_argument('--overwrite',action='store_true')
    check=commands.add_parser('check-data',help='Validate schemas and report aggregate cohort counts')
    check.add_argument('--data',required=True)
    run=commands.add_parser('run',help='Run the complete training/evaluation pipeline')
    run.add_argument('--data',required=True);run.add_argument('--out',required=True)
    run.add_argument('--config',required=True)
    run.add_argument('--models',nargs='+',default=['dipp_label','lightgbm','lr','random'])
    run.add_argument('--no-validation',action='store_true');run.add_argument('--shap',action='store_true')
    run.add_argument('--splits',help='Optional participant allocation CSV: PATNO, seed, split')
    args=parser.parse_args()
    if args.command=='make-example':
        print(json.dumps(generate_example(args.out,args.seed,args.overwrite),indent=2));return
    if args.command=='check-data':
        from . import CLASSES
        data=load_cohort(args.data)
        print(json.dumps(dict(participants=len(data),assays=len(data.assay_ids),synthetic=data.synthetic,
            class_counts={c:int((data.y==i).sum()) for i,c in enumerate(CLASSES)}),indent=2));return
    config=Config.load(args.config)
    from .workflow import run as workflow
    workflow(args.data,args.out,config,args.models,not args.no_validation,args.shap,args.splits)


if __name__=='__main__':main()
