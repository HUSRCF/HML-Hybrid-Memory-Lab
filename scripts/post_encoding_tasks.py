"""Fresh executable SET streams with a common normally encoded fact prefix."""
from dataclasses import asdict
import random
from sas_memory.tasks import Operation, execute

ROLES=('target_current','target_initial','protected_current')
STYLES=('original','entity_first')
TAIL_KINDS=('unrelated','related')
COUNTS=(8,64)


def make_case(seed):
    rng=random.Random(seed);names=rng.sample(list('ABCDEFGHIJKLMNOP'),8)
    target,protected=names[:2];distractors=names[2:];values=rng.sample(range(100,1000),73)
    initial=[Operation('SET',name,value=values[i]) for i,name in enumerate(names)];rng.shuffle(initial)
    prefix=initial+[Operation('SET',target,value=values[8])]
    master=[Operation('SET',rng.choice(distractors),value=v) for v in values[9:]]
    episodes=[{'id':'before','kind':'before','record_count':0,'tail':[]}]
    for count in COUNTS:
        for kind in TAIL_KINDS:
            tail=master[:count].copy()
            if kind=='related':tail[-1]=Operation('SET',target,value=tail[-1].value)
            episodes.append({'id':f'{kind}-{count}','kind':kind,'record_count':count,'tail':[asdict(e) for e in tail]})
    case={'id':f'post-encoding-s{seed}','seed':seed,'target':target,'protected':protected,
        'prefix_events':[asdict(e) for e in prefix],'episodes':episodes}
    validate_case(case)
    return case


def question(case,role,style):
    if role not in ROLES or style not in STYLES:raise ValueError('Explicit role and style required')
    entity=case['protected'] if role=='protected_current' else case['target']
    if style=='original':
        return (f'What was the initial balance of account {entity}?' if role=='target_initial' else
            f'What is the current balance of account {entity}?')
    return (f'For account {entity}, what was its initial balance?' if role=='target_initial' else
        f'For account {entity}, what is its current balance?')


def truth(case,episode,role):
    if role not in ROLES:raise ValueError('Explicit task role required')
    events=[Operation(**e) for e in case['prefix_events']+episode['tail']]
    state=execute(events);entity=case['protected'] if role=='protected_current' else case['target']
    field='initial' if role=='target_initial' else 'current'
    answer=str(state[field][entity])
    candidates=[i for i,e in enumerate(events) if e.target==entity]
    support=candidates[0] if field=='initial' else candidates[-1]
    if events[support].kind!='SET' or str(events[support].value)!=answer:
        raise ValueError('Single-SET evidence and interpreter truth disagree')
    return answer,support


def validate_case(case):
    prefix=[Operation(**e) for e in case['prefix_events']]
    if len(prefix)!=9 or len({e.target for e in prefix})!=8 or any(e.kind!='SET' for e in prefix):
        raise ValueError('Normal nine-record eight-account prefix required')
    target,protected=case['target'],case['protected']
    if target==protected or prefix[-1].target!=target or sum(e.target==target for e in prefix)!=2:
        raise ValueError('Distinct target/current/history and protected account required')
    episodes={e['id']:e for e in case['episodes']}
    if list(episodes)!=['before','unrelated-8','related-8','unrelated-64','related-64']:
        raise ValueError('Complete fixed before/after grid required')
    before=episodes['before'];base={r:truth(case,before,r)[0] for r in ROLES}
    if before['tail'] or len(set(base.values()))!=3:raise ValueError('Three distinct baseline facts required')
    for count in COUNTS:
        unrelated,related=(episodes[f'{k}-{count}'] for k in TAIL_KINDS)
        u,v=[Operation(**x) for x in unrelated['tail']],[Operation(**x) for x in related['tail']]
        if (len(u)!=count or len(v)!=count or u[:-1]!=v[:-1] or
            u[-1].value!=v[-1].value or u[-1].target in (target,protected) or v[-1].target!=target or
            any(e.kind!='SET' or e.target in (target,protected) for e in u) or
            any(e.target in (target,protected) for e in v[:-1])):
            raise ValueError('Related tail must change only the last distractor SET target')
        if any(truth(case,unrelated,r)[0]!=base[r] for r in ROLES):raise ValueError('Unrelated updates changed a queried fact')
        if (truth(case,related,'target_current')[0]==base['target_current'] or
            any(truth(case,related,r)[0]!=base[r] for r in ROLES[1:])):
            raise ValueError('Related update must change current target only')
    return True
