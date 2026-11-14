"""Export every20 audited state-continuation cell with no case selection."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sas_memory.write_locality_data import streaming_hash
from scripts.analyze_state_continuation import audit_run
from scripts.run_controller_trial import write_json


def export(audit_path,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(audit_path.read_text());actual=audit_run(Path(report['source_run']))
    if report!=actual:raise ValueError('Export must reproduce complete independent audit')
    if prefix.with_suffix('.png').exists():raise ValueError('Preserve earlier exports')
    labels=['native_reference','intervened_states','restore_convolution','restore_recurrent','restore_both']
    headings=['Unmasked\nreference','Both paths restored\nstate retained','Restore\nconvolution','Restore\nrecurrent state','Restore\nboth states']
    cells=[];values=[];row_labels=[]
    for case in report['cases']:
        baseline=case['mean_gold_losses']['native_reference'];row=[]
        model={'local_question':'LocalQ','gdn_question_only':'GDNQ'}[case['model']]
        row_labels.append(f"{model} / {case['source_regime']}\noriginal {case['original_direction']} / gold {case['readouts']['native_reference']['answer']}")
        for label in labels:
            readout=case['readouts'][label];delta=case['mean_gold_losses'][label]-baseline
            cells.append({'model':case['model'],'id':case['id'],'condition':label,'readout':readout,
                'mean_gold_nll_difference_vs_native':delta,'gold_losses':case['gold_losses'][label],
                'gold_later_vector_exact_to_native':case['gold_later_vector_exact_to_native'][label]})
            row.append(delta)
        values.append(row)
    if len(cells)!=20:raise ValueError('All four cases and all five conditions required')
    data=np.array(values);bound=max(float(np.abs(data).max()),1e-6)
    fig,ax=plt.subplots(figsize=(12.2,7.6),dpi=180)
    fig.subplots_adjust(left=.24,right=.89,top=.78,bottom=.24)
    im=ax.imshow(data,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(5),headings,fontsize=9);ax.xaxis.tick_top()
    ax.set_yticks(range(4),row_labels,fontsize=9);ax.tick_params(length=0,pad=8)
    for i in range(4):
        for j in range(5):
            cell=cells[i*5+j];readout=cell['readout'];text=readout['generated'].strip() or '(empty)'
            status='correct' if readout['greedy_exact'] else 'wrong'
            ax.text(j,i,f'{text} [{status}]\nNLL {readout["answer_nll"]:.6f}\nchange {data[i,j]:+.6f}',
                ha='center',va='center',fontsize=8.5,color='white' if abs(data[i,j])>.65*bound else 'black')
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.3);ax.tick_params(which='minor',length=0)
    cax=fig.add_axes([.915,.28,.018,.46]);bar=fig.colorbar(im,cax=cax,ticks=[-bound,0,bound])
    bar.ax.tick_params(labelsize=8);bar.set_label('Mean gold NLL change vs reference',fontsize=8)
    fig.suptitle('After exact first-logit closure: do remaining GDN states affect the full answer?',fontsize=13,y=.95)
    fig.text(.24,.88,'Four reused queries; identical complete K/V and starting logits in every condition.',fontsize=10)
    fig.text(.24,.13,'Only conv/recurrent caches in layers20-22 vary. All20 greedy answers and fixed-gold losses shown.\n'
        'Red: higher gold NLL. Blue: lower gold NLL. Color does not measure answer accuracy.\n'
        'Oracle state restoration is a diagnostic; this is not independent confirmation or an architecture result.',fontsize=9,va='top')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height();boxes=[]
    for artist in fig.findobj(match=matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>width or box.y1>height:raise ValueError(f'Figure text outside canvas: {artist.get_text()}')
        boxes.append({'text':artist.get_text(),'bounds':[box.x0,box.y0,box.x1,box.y1]})
    paths=[prefix.with_suffix(s) for s in ('.png','.pdf','.json','.layout.json','.source.py')]
    fig.savefig(paths[0]);fig.savefig(paths[1]);plt.close(fig)
    write_json(paths[2],{'source_audit_sha256':streaming_hash(audit_path),'all20cells_retained':True,'cells':cells})
    write_json(paths[3],{'canvas_pixels':[width,height],'all_text_inside':True,'text_boxes':boxes})
    paths[4].write_bytes(Path(__file__).read_bytes())
    write_json(prefix.with_suffix('.provenance.json'),{'source_audit_sha256':streaming_hash(audit_path),
        'source_manifest_sha256':report['manifest_sha256'],'all20cells_retained':True,
        'artifacts_sha256':{str(p):streaming_hash(p) for p in paths}})
    print(json.dumps({'status':'exported_all20cells','canvas_pixels':[width,height],'artifacts':[str(p) for p in paths]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('audit',type=Path);parser.add_argument('--prefix',type=Path,required=True)
    args=parser.parse_args();export(args.audit,args.prefix)
