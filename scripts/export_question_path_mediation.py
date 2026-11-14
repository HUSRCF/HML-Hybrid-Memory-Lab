"""Render all32audited path-patching observations using a fixed competitor."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_question_path_mediation import audit_run
from scripts.question_path_controls import input_scope
from scripts.run_controller_trial import write_json


def export(audit_path,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    report=json.loads(audit_path.read_text());run=Path(report['source_run'])
    if audit_run(run)!=report:raise ValueError('Independent mediation audit or actual raw vectors changed')
    plan=input_scope()['plan'];contexts=plan['context_order'];patches=plan['patch_order']
    rows=[json.loads(line) for name in plan['models'] for line in (run/f'{name}.jsonl').read_text().splitlines()]
    indexed={(r['model'],r['id'],r['context'],r['patch']):r for r in rows}
    cells=[];matrices=[]
    for context in contexts:
        matrix=[]
        for index,e in enumerate(plan['queries']):
            values=[]
            for patch in patches:
                row=indexed[(e['model'],e['id'],context,patch)];values.append(row['fixed_pair_margin'])
                cells.append({'case_index':index+1,**row})
            matrix.append(values)
        matrices.append(np.asarray(matrix))
    extent=max(max(float(np.abs(m).max()) for m in matrices),1e-9)
    labels=[f"{i+1}. {'Local-Q' if e['model']=='local_question' else 'GDN-Q'} | gold {e['expected_full']['answer']}" for i,e in enumerate(plan['queries'])]
    fig,axes=plt.subplots(2,1,figsize=(11.5,7.8))
    fig.subplots_adjust(left=.24,right=.86,bottom=.18,top=.88,hspace=.48)
    for panel,(ax,matrix,title) in enumerate(zip(axes,matrices,
            ('Unmasked oracle self-to-self controls','Earlier-question Attention19 restriction'))):
        im=ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent),aspect='auto')
        ax.set_xticks(range(4),['None','GDN output','Question KV','Both']);ax.set_yticks(range(4),labels)
        ax.set_title(title,fontsize=11,pad=12)
        for i in range(4):
            for j in range(4):
                row=cells[panel*16+i*4+j];f=row['first_token']
                text=f['argmax_text'].replace('\n','\\n');star=' *' if f['argmax_id']==f['gold_id'] else ''
                ax.text(j,i,f"{text}{star}\n{row['fixed_pair_margin']:+.3f}",ha='center',va='center',fontsize=10,
                    color='white' if abs(matrix[i,j])>.55*extent else 'black')
    axes[-1].set_xlabel('Oracle activation patch; all model weights and native state-write kernels frozen')
    cax=fig.add_axes([.89,.30,.018,.46]);bar=fig.colorbar(im,cax=cax);bar.set_ticks(np.linspace(-extent,extent,5))
    cax.set_ylabel('Gold first-token logit minus registered fixed competitor')
    fig.suptitle('GDN current output versus earlier-question K/V routes',fontsize=14,y=.97)
    fig.text(.24,.105,'Cell text: first-token argmax and fixed-pair margin. * Gold first token; full answers are not scored.',fontsize=8)
    fig.text(.24,.067,'Oracle closure restores the reference computation, which can itself be wrong. All 32 cells retained.',fontsize=8)
    fig.text(.24,.032,'Posthoc computational mediation; output patches do not restore recurrent state or establish an architecture benefit.',fontsize=8)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;checked=0
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            box=item.get_window_extent(renderer)
            if not (bounds.x0<=box.x0 and bounds.y0<=box.y0 and box.x1<=bounds.x1 and box.y1<=bounds.y1):
                raise ValueError(f'Figure text outside canvas: {item.get_text()}')
            checked+=1
    paths=[Path(str(prefix)+ext) for ext in ('.png','.pdf','.json','.layout.json','.source.py')]
    provenance=Path(str(prefix)+'.provenance.json')
    if any(p.exists() for p in [*paths,provenance]):raise ValueError('Preserve prior exports')
    fig.savefig(paths[0],dpi=180);fig.savefig(paths[1]);plt.close(fig)
    write_json(paths[2],{'posthoc':True,'independent_scientific_confirmation':False,'source_audit':str(audit_path),
        'source_audit_sha256':streaming_hash(audit_path),'all32cells':cells,'fixed_pair_margin_range':[-extent,extent]})
    write_json(paths[3],{'all_text_inside_canvas':True,'checked_text_objects':checked,'physical_cells':32,
        'panels':2,'rows_per_panel':4,'columns':4,'source_manifest_sha256':report['manifest_sha256']})
    paths[4].write_bytes(Path(__file__).read_bytes())
    write_json(provenance,{'source_audit_sha256':streaming_hash(audit_path),'source_manifest_sha256':report['manifest_sha256'],
        'all32cells_retained':True,'artifacts_sha256':{str(p):streaming_hash(p) for p in paths}})
    print(json.dumps({'exported_artifacts':len(paths),'cells':32,'all_text_inside_canvas':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();export(args.audit,args.prefix)
