"""New-contract CLI, immutable bundles and replay integrity."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hw_model.config import load_config
from hw_model.experiments import ROOT, MAPPINGS, OPERATORS, compare, replay, dump, main, arithmetic_checks


class Arguments:
    pass


def mock_run(config,capture_trace=False):
    rows=[{'mapping':mapping,'architecture':'array_9x16' if mapping=='A' else 'row_dot128',
           'operator':operator,'total_cycles':100,'last_internal_output_confirm':100}
          for operator in OPERATORS for mapping in MAPPINGS]
    return rows,[]


class ExperimentTests(unittest.TestCase):
    def test_fixed300_l0_checks_replace_old_ln64_example(self):
        result=arithmetic_checks()
        fixed=result['fixed300_weight_dominated_l0']
        totals=[item[mapping]['total_cycles'] for item in fixed['operators'] for mapping in ('A','B0')]
        self.assertEqual(totals,[23680,202880,91264,811136])
        self.assertEqual(len(fixed['weight_ledger']),8)

    def test_cli_rejects_removed_sweep_options(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as raised:
            main(['compare','--sweep','bandwidth'])
        self.assertEqual(raised.exception.code,2)

    def test_saved_eight_results_and_sources_replay_exactly(self):
        reports=ROOT/'hw_model'/'reports'
        reports.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(reports)) as temporary:
            base=Path(temporary)
            config=load_config(overrides={'model':{'tokens':1,'ffn':64}})
            config_path=base/'input.json'
            dump(config_path,config)
            args=Arguments()
            args.config,args.out=str(config_path),str(base/'experiment')
            args.trace,args.no_trace=False,True
            with patch('hw_model.experiments.run_scenarios',side_effect=mock_run),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare(args),0)
                output=Path(args.out)
                summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['schema_version'],2)
                self.assertEqual(len(summary['results']),8)
                self.assertNotIn('sweep',summary)
                self.assertTrue((output/'source'/'hw_model'/'config.py').is_file())
                self.assertTrue((output/'report.html').is_file())
                self.assertFalse((output/'sweep_configs.json').exists())
                again=Arguments()
                again.report=args.out
                self.assertEqual(replay(again),0)
                edited=load_config(config_path)
                edited['provenance']['scenario_id']='edited'
                dump(output/'run_config.json',edited)
                with self.assertRaisesRegex(ValueError,'hash'):
                    replay(again)
                with self.assertRaises(FileExistsError):
                    compare(args)

    def test_old_report_requires_its_frozen_source(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as temporary:
            dump(Path(temporary)/'summary.json',{'schema_version':1})
            args=Arguments()
            args.report=temporary
            with self.assertRaisesRegex(ValueError,'legacy'):
                replay(args)


if __name__=='__main__':
    unittest.main()
