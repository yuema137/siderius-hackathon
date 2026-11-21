"""Offline guard tests; these do not certify native APIs or model calls."""
import importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('tess_preview',ROOT/'preview.py');preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(preview)
class GuardTests(unittest.TestCase):
 def test_offline_missing_data_refuses_without_network(self):
  with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen',side_effect=AssertionError('network called')):
   with self.assertRaisesRegex(ValueError,'Missing cached'):preview.fetch(Path(td),offline=True)
 def test_modified_cached_file_refuses(self):
  with tempfile.TemporaryDirectory() as td, patch('urllib.request.urlopen',side_effect=AssertionError('network called')):
   (Path(td)/'identity.csv').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'checksum mismatch'):preview.fetch(Path(td),offline=True)
 def test_missing_key_stops_before_framework_or_network(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p=root/'preview';p.mkdir();(p/'baseline.json').write_text(json.dumps({'exp_revision':preview.EXP_REV,'validation_count':442}))
   env=dict(os.environ);env.pop('OPENAI_API_KEY',None)
   result=subprocess.run([sys.executable,str(ROOT/'orchestrate.py'),'--preview-dir',str(p),'--output-dir',str(root/'live'),'--live'],env=env,capture_output=True,text=True)
   self.assertEqual(result.returncode,2);status=json.loads((root/'live/status.json').read_text());self.assertEqual(status['state'],'failed');self.assertEqual(status['bridge_calls_started'],0);self.assertIn('OPENAI_API_KEY is absent',result.stdout)
if __name__=='__main__':unittest.main()
