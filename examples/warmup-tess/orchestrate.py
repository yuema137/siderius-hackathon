#!/usr/bin/env python3
"""Bounded native Interpret → Propose caller. No implementation, training or scoring."""
import argparse, json, os, signal, subprocess, time
from pathlib import Path
from preview import EXP_REV


def bounded_bridge_factory(inner_factory, status, save, out, *, clock=time.monotonic, evidence_context=None):
 started=clock()
 class BoundedBridge:
  def __init__(self,**kwargs):
   kwargs.update(provider='openai',model_id='gpt-6-luna',reasoning_effort='low',max_retries=0,timeout_retries=0,request_timeout=55,max_output_tokens=8192,reflect_max_output_tokens=8192)
   self.inner=inner_factory(**kwargs)
   self.inner.set_run_context(workspace=out,iter=0,run_name='warmup',run_id=out.name)
  def __getattr__(self,name):return getattr(self.inner,name)
  def invoke(self,method,*a,**kw):
   # Native legacy commit otherwise receives only generated intermediate reasoning.
   # Carry original preview evidence explicitly; never label it formal/Health-valid.
   if kw.get('label')=='proposer.legacy_commit' and evidence_context:
    context=evidence_context()
    if not context:raise RuntimeError('Missing original preview evidence; refusing a context-free proposal.')
    if len(a)<2:raise RuntimeError('Unexpected native prompt signature.')
    # v1.1.0 legacy commit retains a denoising-specific literal even for [B,1].
    # This caller is TESS-only: correct that literal to the declared scalar target.
    system=a[0].replace('the head emits one continuous value per time step',
      'the head emits one near-core rotation frequency scalar per input light curve, not per time step')
    a=(system,a[1]+'\n\n'+context,*a[2:])
   if status['bridge_calls_started']>=4 or clock()-started>=240:raise RuntimeError('Shared warm-up request/time limit reached.')
   if sum(len(v) for v in (*a,*kw.values()) if isinstance(v,str))>100000:raise RuntimeError('Prompt exceeds warm-up 100,000-character ceiling.')
   status['bridge_calls_started']+=1;save()
   result=getattr(self.inner,method)(*a,**kw)
   if method=='generate_text' and (not isinstance(result,str) or not result.strip()):
    raise RuntimeError('Native reasoning returned no text; refusing a context-free proposal.')
   return result
  def generate(self,*a,**kw):return self.invoke('generate',*a,**kw)
  def generate_text(self,*a,**kw):return self.invoke('generate_text',*a,**kw)
 return BoundedBridge


def preview_evidence_context(baseline, interpreted):
 return ('## Original CPU preview evidence (not formal training or HealthGate evidence)\n'
  +json.dumps({'measured_preview':{k:baseline[k] for k in ['prediction','train_count','validation_count','r2','rmse','mae']},
   'method':'Predict the training-target mean for all validation curves; ignores light curves.',
   'limits':'No native training, no Health verdict, no tested learned improvement. Negative formal evidence is not absence of this preview.',
   'llm_interpretation_not_independently_verified':{'key_findings':interpreted.key_findings,'bottlenecks':interpreted.bottlenecks,'take_home_message':interpreted.take_home_message}},indent=2)
  +'\nUse the measured baseline in the motivation. Distinguish interpretation hypotheses from observed facts. Propose one comparison; do not claim success before testing.')


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--preview-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
 ap.add_argument('--siderius-exp',type=Path,help='Optional existing checkout at the pinned revision; otherwise fetched automatically.')
 ap.add_argument('--live',action='store_true',help='Authorize bounded real Luna API calls. Without this, only validate native input.')
 args=ap.parse_args();preview=args.preview_dir.resolve();out=args.output_dir.resolve()
 baseline=json.loads((preview/'baseline.json').read_text())
 if baseline.get('exp_revision')!=EXP_REV or baseline.get('validation_count')!=442:raise SystemExit('Run the matching preview first.')
 out.mkdir(parents=True,exist_ok=False)
 status={'state':'preparing','live_requested':args.live,'bridge_calls_started':0,'native_training':'not_run','wall_limit_seconds':240,'request_limit':4,'output_tokens_per_request':8192}
 def save(): (out/'status.json').write_text(json.dumps(status,indent=2))
 save()
 try:
  if args.live and not os.environ.get('OPENAI_API_KEY'):
   raise RuntimeError('OPENAI_API_KEY is not set. No key value was read or printed. No API call started.')
  # Imports must come from this checkout's installed, frozen SIDERIUS environment.
  from importlib.metadata import version
  if version('siderius')!='1.1.0':raise RuntimeError('This caller targets SIDERIUS v1.1.0; use the documented release.')
  from core.generated_library import bind_generated_library_to_workspace
  bind_generated_library_to_workspace(str(out))
  from core.local_code import root_code_scope
  exp=args.siderius_exp.resolve() if args.siderius_exp else out/'official-task-source'
  if not args.siderius_exp:
   subprocess.run(['git','init','-q',str(exp)],check=True)
   subprocess.run(['git','-C',str(exp),'remote','add','origin','https://github.com/yuema137/siderius-exp.git'],check=True)
   subprocess.run(['git','-C',str(exp),'fetch','-q','--depth','1','origin',EXP_REV],check=True,timeout=60)
   subprocess.run(['git','-C',str(exp),'checkout','-q','--detach','FETCH_HEAD'],check=True)
  if subprocess.check_output(['git','-C',str(exp),'rev-parse','HEAD'],text=True).strip()!=EXP_REV:
   raise RuntimeError('The supplied task checkout is not the pinned revision.')
  with root_code_scope():
   from workflows.task_composition import compose_run_task_bindings, bind_run_task_composition, build_task_composition_ref
   from agent.schemas.interpretation import InterpretationInput, ModelRunSummary
   from agent.schemas.storage import StorageConfig, LocalStorageConfig
   from ml_models.model_descriptions import DescriptionSourcePolicy
   composition=compose_run_task_bindings(str(exp/'tasks/phyts_tess/compositions/rotation_regression.yaml'))
   with bind_run_task_composition(composition,physical_data_root=str(preview/'data')):
    storage=StorageConfig(local=LocalStorageConfig(workspace=str(out),run_name='warmup'))
    summary=ModelRunSummary(model_type='training_mean',run_name='cpu_preview',status='completed',completed_rounds=1,best_denoising_score=baseline['r2'],worst_denoising_score=baseline['r2'],round_scores=[baseline['r2']],best_config={'constant_frequency':baseline['prediction']},model_description='A deterministic constant baseline: predict the mean of all 3338 training rotation targets for each of 442 validation curves. No neural-network optimization, no native tuner, no formal Health verdict. Measured CPU preview, not a competition submission.')
    request=InterpretationInput(summaries=[summary],cold_start=False,task_description=composition.task_description,metric_spec=composition.selection_metric_spec,task_blocks=composition.interpretation_blocks,description_source_policy=DescriptionSourcePolicy.COMPOSED,storage=storage)
    (out/'interpretation-input.json').write_text(request.model_dump_json(indent=2))
    (out/'request-check.json').write_text(json.dumps({'task_source':str(exp/'tasks/phyts_tess/compositions/rotation_regression.yaml'),'task_revision':EXP_REV,'baseline_source':str(preview/'baseline.json'),'cold_start':False,'observed_validation_population':442,'metric_id':composition.selection_metric_spec.id,'no_training_or_data_materialization_enabled':True,'interpretation_input_validated':True},indent=2))
    if not args.live:
     status['state']='native_input_validated_no_api';save();print('Native interpretation input validated. No Luna call, no training.');return
    # Supported dependency injection sets stricter transport limits without replacing native agents.
    from agent.llm_bridge import LLMBridge
    from nodes.result_interpretation_agent.result_interpretation_agent import ResultInterpretationAgent
    from nodes.ml_model_proposal_agent.ml_model_proposal_agent import MLModelProposalAgent
    from agent.schemas.protocols.ml_result_interp_to_ml_model_propose import local_full_context
    from agent.schemas.proposal import ProposalInput
    def timeout_handler(signum,frame):raise TimeoutError('Shared 240-second API-call wall budget exhausted.')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(240)
    context_holder={}
    BoundedBridge=bounded_bridge_factory(LLMBridge,status,save,out,evidence_context=lambda:context_holder.get('evidence'))
    status['state']='interpretation_started';save()
    interpreted=ResultInterpretationAgent(provider='openai',model_id='gpt-6-luna',max_retries=0,reasoning_effort='low',bridge_factory=BoundedBridge).run(request)
    (out/'interpretation.json').write_text(interpreted.model_dump_json(indent=2))
    context_holder['evidence']=preview_evidence_context(baseline,interpreted)
    (out/'proposal-evidence-context.txt').write_text(context_holder['evidence'])
    reference=build_task_composition_ref(composition)
    projected=local_full_context(interpreted,storage,task_composition_ref=reference)
    payload=projected.model_dump();payload.update(task_description=composition.task_description,forward_contract=composition.forward_contract,task_composition_ref=reference,constraints=['Warm-up only. Propose one small CPU-friendly model that uses the light curve. Do not claim it was implemented or trained.','Keep the declared star-grouped training/validation split and R-squared metric. No external retrieval, no new data, no test split.'],formal_time_budget_minutes=None,trial_time_budget_minutes=None)
    proposal_input=ProposalInput.model_validate(payload)
    (out/'proposal-input.json').write_text(proposal_input.model_dump_json(indent=2))
    status['state']='proposal_started';save()
    proposed=MLModelProposalAgent(provider='openai',model_id='gpt-6-luna',max_retries=0,reasoning_effort='low',bridge_factory=BoundedBridge).run(proposal_input)
    (out/'proposal.json').write_text(proposed.model_dump_json(indent=2));signal.alarm(0)
    status['state']='native_interpretation_and_proposal_returned';status['scientific_content_review']='required_not_automatically_certified';save()
    (out/'READ-ME.txt').write_text('The native agents returned interpretation.json and proposal.json. These are advice/proposal artifacts, not implemented models, training evidence or a measured improvement. Inspect status and diagnostics in both outputs. Compare the proposal with the measured baseline.json; run formal experiments later on the H100.\n')
    print('Returned native interpretation and proposal. No model was trained. See',out)
 except Exception as exc:
  signal.alarm(0);status.update(state='failed',error_type=type(exc).__name__)
  # Avoid copying provider exception bodies, which may carry request context, into public artifacts.
  save();print(f'Warm-up did not complete: {type(exc).__name__}. See status.json; no success is claimed.')
  if isinstance(exc,ModuleNotFoundError):print('Use the selected SIDERIUS checkout’s frozen .venv/bin/python after uv sync --group dev --frozen.')
  elif isinstance(exc,RuntimeError) and not os.environ.get('OPENAI_API_KEY'):print('OPENAI_API_KEY is absent; no API call was made.')
  raise SystemExit(2)
if __name__=='__main__':main()
