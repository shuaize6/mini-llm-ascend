"""M3 短正确性验收，使用 tokenizer 输出的短前缀；不测吞吐。"""

import argparse
from dataclasses import asdict

from bench.common import open_device, write_report, environment_info
from bench.m1_common import prepare_input, provenance, compare_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', choices=('cpu','npu'), default='npu')
    parser.add_argument('--dtype', choices=('float16','float32'), default='float16')
    parser.add_argument('--output', default='results/m3.json')
    args = parser.parse_args()
    report = {'args': vars(args), 'environment': environment_info(), 'status': 'running',
              'passed': False, 'scope': 'short token prefixes, correctness only; no throughput claim'}
    scheduler = None
    def save(stage):
        report['stage'] = stage
        write_report(args.output,report)
        print(stage,flush=True)
    try:
        torch, device, dtype = open_device(args.device,args.dtype)
        from mini_llm_ascend.engine.weights import load_model
        from mini_llm_ascend.engine.model_runner import ModelRunner
        from mini_llm_ascend.engine.batch_runner import forward_batch
        from mini_llm_ascend.engine.scheduler import Scheduler
        from mini_llm_ascend.cache.paged_kv import PagedKVPool
        save('loading')
        report['provenance'] = provenance(args.model)
        _, ids, _, eos = prepare_input(args.model,'请解释KV缓存。',device)
        model = load_model(args.model,device=device,dtype=dtype)
        pool = PagedKVPool.for_model(model,8,4)
        scheduler = Scheduler(model,pool,2)
        inputs = [ids[:,:2].clone(),ids[:,:1].clone()]
        report['input_ids'] = [x[0].cpu().tolist() for x in inputs]
        save('logits_comparison')
        caches = [pool.new_cache(8) for _ in inputs]
        refs = [model.new_cache(8) for _ in inputs]
        report['comparisons'] = []
        try:
            for name, batch in [('prefill',inputs),('mixed', [inputs[1],inputs[0]])]:
                expected = torch.cat([model(x,c) for x,c in zip(batch,refs)])
                actual = forward_batch(model,batch,caches)
                result = compare_tensors(actual,expected,atol=0.01 if dtype==torch.float16 else 1e-5,
                                         rtol=0.01 if dtype==torch.float16 else 1e-4,logits=True)
                report['comparisons'].append({'name':name, 'comparison':result})
                if not result['passed']: raise RuntimeError('batch logits mismatch')
        finally:
            for cache in caches: cache.close()
        save('dynamic_requests')
        a = scheduler.submit('a',inputs[0],3,eos_token_id=eos,fixed_length=True)
        report['steps'] = [[asdict(e) for e in scheduler.step()]]
        scheduler.submit('b',inputs[1],1,eos_token_id=eos,fixed_length=True)
        scheduler.submit('c',inputs[0],2,eos_token_id=eos,fixed_length=True)
        while scheduler.has_pending:
            report['steps'].append([asdict(e) for e in scheduler.step()])
        report['requests'] = {}
        save('serial_reference')
        for name,r in scheduler.requests.items():
            expected = ModelRunner(model,8).generate(r.input_ids,r.max_new_tokens,
                                                      eos_token_id=eos,fixed_length=True)
            report['requests'][name] = {'actual':r.output_ids,'expected':expected,'status':r.status}
            if r.output_ids != expected: raise RuntimeError('output IDs mismatch: '+name)
        scheduler.close()
        report['pool_after_close'] = pool.snapshot()
        if report['pool_after_close']['request_handles'] or report['pool_after_close']['allocated_blocks']:
            raise RuntimeError('cache leak')
        report.update(status='ok',passed=True)
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='error',error=repr(exc))
        code = 1
    finally:
        if scheduler is not None: scheduler.close()
    save('complete' if report['passed'] else 'failed')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
