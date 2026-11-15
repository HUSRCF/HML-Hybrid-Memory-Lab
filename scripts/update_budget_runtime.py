"""Keep write selection independent of the original semantic evidence reader."""
from contextlib import ExitStack
from scripts.gate_component_runtime import ComponentHold,GDN_LAYERS,component_kernel_contract,validation_addendum,VALIDATION_ADDENDUM_SHA256
from scripts.update_budget_data import FAMILIES
from sas_memory.training_data import data_fingerprint

SETTINGS={'full':('original',False,'none'),'oracle':('original',True,'none'),
    'delta_full':('original',False,'delta'),'delta_oracle':('original',True,'delta'),
    **{f'{f}_{r}':(f,r=='oracle','delta') for f in FAMILIES for r in ('full','oracle')}}

def measure_condition(runtime,ledger,query,policy,regime,observe=False):
    from sas_memory.kv_readout import QuestionKVReadout
    from scripts.copy_readout_runtime import SelectionWeights,check_weight_stats
    from scripts.run_oracle_kv_readout import measure
    from scripts.run_write_locality_load_screen import hook_state
    family,restricted,mode=SETTINGS[regime];reader_bits=policy['history_keep_bits']
    write_bits=reader_bits if family=='original' else policy['write_keep_bits'][family]
    if len(write_bits)!=len(reader_bits) or write_bits.count('0')!=reader_bits.count('0'):raise ValueError('Exact original allowed-write budget required')
    write_keep=[b=='1' for b in write_bits];reader_keep=[b=='1' for b in reader_bits];active=hook_state(runtime.model)
    with ExitStack() as stack:
        hold=stack.enter_context(ComponentHold(runtime.model,write_keep,mode))
        reader=stack.enter_context(QuestionKVReadout(runtime.model,reader_keep)) if restricted else None
        observer=stack.enter_context(SelectionWeights(runtime.model,{'actual':reader_keep},query['prelude_tokens'])) if observe else None
        value=measure(runtime,ledger,query);gates=hold.report()
        if observer:check_weight_stats(observer.stats['actual'],restricted)
        stats=observer.stats['actual'] if observer else None
    if hook_state(runtime.model)!=active:raise ValueError('Independent writer/reader/observer cleanup failed')
    return value,{'write_family':family,'write_keep_sha256':data_fingerprint(write_bits),
        'reader_keep_sha256':data_fingerprint(reader_bits),'gates':gates,
        'reader_calls':reader.calls if reader else None,'attention':stats}
