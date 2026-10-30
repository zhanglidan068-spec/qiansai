"""Fixed-contract validation and honest resource/report accounting."""
import json
from pathlib import Path
import tempfile
import unittest
from hw_model.config import CONFIG_DIR, load_config, source_manifest, validate_config
from hw_model.compute import Shape
from hw_model.resources import resource_inventory
from hw_model.reporting import write_report


ROOT=Path(__file__).absolute().parents[2]


class ConfigResourceTests(unittest.TestCase):
    def test_fixed_contract_resolved_and_independent(self):
        first,second=load_config(),load_config()
        first['model']['tokens']=1
        self.assertEqual(second['model']['tokens'],99)
        self.assertEqual(second['schema_version'],2)
        self.assertEqual(second['platform']['weight_bytes_per_second'],300000000)
        self.assertEqual(second['numeric']['output_bits'],8)

    def test_contract_changes_cannot_silently_become_sweeps(self):
        changes=[('platform','weight_beat_bytes',16),('platform','weight_accept_interval',1),
                 ('platform','weight_bytes_per_second',400000000),('ln','latency_cycles',64),
                 ('ln','initiation_interval_cycles',1),('ln','max_inflight_rows',2),
                 ('local_memory','banks',64),('local_memory','byte_write_mask',False),
                 ('numeric','output_bits',16),('architecture','post_channels_per_cycle',16)]
        for section,key,value in changes:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'fixed'):
                load_config(overrides={section:{key:value}})

    def test_missing_unknown_and_legacy_fields_rejected(self):
        for patch in ({'architecture':{'weight_cache':'whole_matrix'}},
                      {'platform':{'beat_bytes':4}},
                      {'architecture':{'weight_banks':8}},
                      {'sweep':'bandwidth'}):
            with self.subTest(patch=patch),self.assertRaisesRegex(ValueError,'unknown'):
                load_config(overrides=patch)
        config=load_config()
        del config['ln']['latency_cycles']
        with self.assertRaisesRegex(ValueError,'missing'):
            validate_config(config)
        with self.assertRaises(ValueError):
            load_config(CONFIG_DIR.parent/'spec'/'historical_v1_configs'/'platform_unknown.json')

    def test_pipeline_and_reduced_shape_test_parameters(self):
        config=load_config(overrides={'model':{'tokens':1,'ffn':7},
                                      'architecture':{'multiply_latency':3,'accumulator_feedback':2}})
        self.assertEqual(config['model']['ffn'],7)
        self.assertEqual(config['architecture']['multiply_latency'],3)
        for patch in ({'model':{'tokens':0}},{'model':{'hidden':32}},
                      {'architecture':{'multiply_latency':0}},
                      {'architecture':{'accumulator_feedback':True}}):
            with self.assertRaises(ValueError):
                load_config(overrides=patch)

    def test_fixed_sram_and_small_registers_counted_separately(self):
        config=load_config()
        for mapping,weight in [('A',32),('B0',256),('B1_G9',256),('B1_G99',256)]:
            result=resource_inventory(config,mapping,Shape(99,64,256))
            self.assertEqual(result['macro_instances'],{'4096_single':8})
            self.assertEqual(result['local_sram_bytes'],65536)
            self.assertEqual(result['weight_sram_bytes'],0)
            self.assertEqual(result['register_payload_bytes']['weight_register_bytes'],weight)
            self.assertEqual(result['register_payload_bytes']['parameter_register_bytes'],512)
            self.assertIsNone(result['area']['total_mm2'])
            self.assertIsNone(result['area']['within_limit'])
        result=resource_inventory(config,'B1_G99',Shape(99,64,256))
        self.assertEqual(result['b1_partial_sram_bytes'],792)
        self.assertEqual(result['common_tensor_payload_bytes'],50688)
        self.assertNotIn('b1_partial_sram_bytes',result['register_payload_bytes'])

    def test_measured_registers_override_and_partials_do_not_duplicate(self):
        result=resource_inventory(load_config(),'B1_G99',(99,64,256),
                                  {'activation_register_bytes':64,'b1_partial_sram_bytes':792,
                                   'pipeline_register_bytes':16})
        self.assertEqual(result['register_payload_bytes']['activation_register_bytes'],64)
        self.assertEqual(result['register_payload_bytes']['pipeline_register_bytes'],16)
        self.assertEqual(result['total_register_payload_bytes'],sum(result['register_payload_bytes'].values()))
        self.assertNotIn('b1_partial_sram_bytes',result['register_payload_bytes'])

    def test_provided_area_stays_an_estimate(self):
        config=load_config(overrides={'area':{'standard_cells_mm2':0.1,'macro_areas_mm2':{'4096_single':0.02}}})
        result=resource_inventory(config,'A',(99,64,256))
        self.assertAlmostEqual(result['area']['total_mm2'],0.26)
        self.assertEqual(result['area']['status'],'provided_estimate_not_synthesis')

    def test_manifest_excludes_report_outputs(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            root=Path(tmp)
            (root/'hw_model'/'reports').mkdir(parents=True)
            (root/'hw_model'/'one.py').write_text('x=1\n',encoding='utf-8')
            first=source_manifest(root)
            (root/'hw_model'/'reports'/'summary.json').write_text('{}',encoding='utf-8')
            self.assertEqual(first,source_manifest(root))
            self.assertEqual(list(first['files']),['hw_model/one.py'])

    def test_report_has_new_boundary_no_sweep_and_complete_counters(self):
        config=load_config()
        result={'mapping':'A','architecture':'array_9x16','operator':'projection','total_cycles':99,
                'first_matrix_fragment':5,'first_matrix_row':20,'first_ln_row_done':80,
                'last_internal_output_confirm':99,'issue_lane_utilization':1.0,'issue_duty_cycle':.1,
                'compute_utilization':.1,'weight_port':{'useful_weight_bytes':45056,'physical_weight_bytes':45056},
                'local_memory':{'per_bank':[{'reads':7}]},'buffers':{'row':{'peak_live_bytes':128}},
                'initialization':{'external_cycles':None},'resources':resource_inventory(config,'A',(99,64,64))}
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmp:
            write_report(tmp,config,[result],{'aggregate_sha256':'abc'},[])
            path=Path(tmp)
            summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['results'],[result])
            self.assertFalse(summary['claims']['main_window_includes_shared_output_writeback'])
            self.assertNotIn('sweep',summary)
            self.assertTrue((path/'initialization.csv').is_file())
            self.assertFalse((path/'sweep.csv').exists())
            viewer=(path/'report.html').read_text(encoding='utf-8')
            self.assertIn('最后内部输出确认',viewer)
            self.assertIn('不是系统冷启动时间',viewer)
            self.assertNotIn('<script src=',viewer)
            with self.assertRaisesRegex(ValueError,'sweeps'):
                write_report(tmp,config,[],{},[],[{'axis':'bandwidth'}])


if __name__=='__main__':
    unittest.main()
