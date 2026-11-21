"""Native schema/handoff and mocked transport guards. No API or training calls."""
import json, tempfile, unittest
from pathlib import Path
from orchestrate import bounded_bridge_factory

class FakeTransport:
 def __init__(self,**kwargs):self.options=kwargs;self.calls=[]
 def set_run_context(self,**kwargs):self.context=kwargs
 def generate(self,*a,**kw):self.calls.append((a,kw));return 'mock response, not Luna'
 generate_text=generate

class BudgetTests(unittest.TestCase):
 def setup_bridge(self):
  self.now=0;self.saved=[];self.status={'bridge_calls_started':0}
  factory=bounded_bridge_factory(FakeTransport,self.status,lambda:self.saved.append(dict(self.status)),Path('/tmp/test-only'),clock=lambda:self.now)
  return factory(),factory()
 def test_shared_request_limit_and_routing(self):
  a,b=self.setup_bridge()
  for item in (a,b,a,b):item.generate('prompt')
  with self.assertRaises(RuntimeError):a.generate_text('fifth')
  self.assertEqual(self.status['bridge_calls_started'],4)
  self.assertEqual(len(a.inner.calls)+len(b.inner.calls),4)
  self.assertEqual(a.inner.options['model_id'],'gpt-6-luna')
  self.assertEqual(a.inner.options['reasoning_effort'],'low')
  for key,value in {'max_retries':0,'timeout_retries':0,'request_timeout':55,'max_output_tokens':8192,'reflect_max_output_tokens':8192}.items():self.assertEqual(a.inner.options[key],value)
 def test_wall_limit_before_transport(self):
  a,b=self.setup_bridge();self.now=240
  with self.assertRaises(RuntimeError):b.generate('prompt')
  self.assertEqual(self.status['bridge_calls_started'],0)
 def test_prompt_limit_before_transport(self):
  a,b=self.setup_bridge()
  with self.assertRaises(RuntimeError):a.generate('x'*100001)
  self.assertEqual(self.status['bridge_calls_started'],0)

class EvidenceContinuityTests(unittest.TestCase):
 def test_original_evidence_reaches_native_commit_even_if_reasoning_loses_it(self):
  from orchestrate import preview_evidence_context
  from types import SimpleNamespace
  from nodes.ml_model_proposal_agent.ml_model_proposal_agent import _build_commit_prompt
  baseline={'prediction':1.08968,'train_count':3338,'validation_count':442,'r2':-.009278,'rmse':.56444,'mae':.44647}
  interpreted=SimpleNamespace(key_findings=['Measured baseline ignores input'],bottlenecks=['No light-curve features'],take_home_message='Compare a model using light curves.')
  context=preview_evidence_context(baseline,interpreted)
  state={'bridge_calls_started':0}
  bridge=bounded_bridge_factory(FakeTransport,state,lambda:None,Path('/tmp/test-only'),evidence_context=lambda:context)()
  native_commit=_build_commit_prompt('No interpretation was provided.', ['training_mean'])
  bridge.generate('the head emits one continuous value per time step',native_commit,label='proposer.legacy_commit')
  self.assertIn('scalar per input light curve',bridge.inner.calls[0][0][0])
  self.assertNotIn('one continuous value per time step',bridge.inner.calls[0][0][0])
  prompt=bridge.inner.calls[0][0][1]
  for expected in ['3338','442','-0.009278','No light-curve features','not formal training','No native training']:
   self.assertIn(expected,prompt)
  self.assertEqual(state['bridge_calls_started'],1)
 def test_empty_reasoning_fails_before_commit(self):
  class Empty(FakeTransport):
   def generate_text(self,*a,**kw):return ''
  state={'bridge_calls_started':0}
  bridge=bounded_bridge_factory(Empty,state,lambda:None,Path('/tmp/test-only'))()
  with self.assertRaisesRegex(RuntimeError,'no text'):bridge.generate_text('system','prompt')
  self.assertEqual(state['bridge_calls_started'],1)

class NativeHandoffTests(unittest.TestCase):
 def test_real_native_schema_and_protocol_with_mock_interpretation(self):
  from core.generated_library import bind_generated_library_to_workspace
  with tempfile.TemporaryDirectory() as td:
   bind_generated_library_to_workspace(td)
   from core.local_code import root_code_scope
   with root_code_scope():
    from agent.schemas.interpretation import InterpretationOutput
    from agent.schemas.storage import StorageConfig,LocalStorageConfig
    from agent.schemas.protocols.ml_result_interp_to_ml_model_propose import local_full_context
    from agent.schemas.proposal import ProposalInput
    # Explicit synthetic boundary fixture; never represented as model output.
    result=InterpretationOutput(model_types=['training_mean'],model_descriptions={'training_mean':'Constant mean baseline'},total_experiments=1,key_findings=['Mock fixture: baseline ignores the light curve.'],bottlenecks=['No input features'],take_home_message='Mock fixture: compare a model using light-curve features.',per_model_best={'training_mean':-.009278},per_model_worst={'training_mean':-.009278},best_denoising_score=-.009278,worst_denoising_score=-.009278)
    storage=StorageConfig(local=LocalStorageConfig(workspace=td,run_name='offline-test'))
    projected=local_full_context(result,storage)
    payload=projected.model_dump();payload.update(formal_time_budget_minutes=None,trial_time_budget_minutes=None)
    request=ProposalInput.model_validate(payload)
    self.assertIsNone(request.formal_time_budget_minutes)
    self.assertIsNone(request.trial_time_budget_minutes)
    self.assertIn('training_mean',request.model_dump_json())
if __name__=='__main__':unittest.main()
