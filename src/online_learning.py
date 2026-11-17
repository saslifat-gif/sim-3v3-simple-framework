"""Small on-policy actor-critic updates for 20-wall-second rollout chunks."""
import copy
import math


def update_chunk(model, rows, bootstrap_features=None):
    from .rl_policy import probabilities
    if len(rows)<4 and not (rows and rows[-1]['done']):raise ValueError('Need four fresh transitions or a terminal tail')
    weights=model['weights']; value=model.get('value_weights',[0.]*12)
    future=sum(a*b for a,b in zip(value,bootstrap_features)) if bootstrap_features is not None else 0.
    returns=[]
    for row in reversed(rows):
        future=row['reward']+.99**max(.01,row['duration_sim_seconds']/2)*(0 if row['done'] else future)
        returns.append(future)
    returns.reverse()
    advantages=[target-sum(a*b for a,b in zip(value,r['features'])) for target,r in zip(returns,rows)]
    mean=sum(advantages)/len(rows)
    std=max(.1,math.sqrt(sum((a-mean)**2 for a in advantages)/len(rows)))
    advantages=[(a-mean)/std for a in advantages]
    gradient=[[0.]*12 for _ in range(5)]
    value_gradient=[0.]*12
    for row,target,adv in zip(rows,returns,advantages):
        x=row['features'];p=probabilities(weights,x)
        if any(abs(a-b)>1e-7 for a,b in zip(p,row['probabilities'])):raise ValueError('Stale on-policy rollout')
        entropy=-sum(a*math.log(max(a,1e-12)) for a in p)
        for i in range(5):
            coefficient=((1 if i==row['action'] else 0)-p[i])*adv-.01*p[i]*(math.log(max(p[i],1e-12))+entropy)
            for j in range(12):gradient[i][j]+=coefficient*x[j]/len(rows)
        error=max(-10,min(10,target-sum(a*b for a,b in zip(value,x))))
        for j in range(12):value_gradient[j]+=error*x[j]/len(rows)
    norm=math.sqrt(sum(v*v for row in gradient for v in row));scale=min(1,5/max(norm,1e-12))
    step=.3
    for _ in range(20):
        proposed=[[weights[i][j]+step*scale*gradient[i][j] for j in range(12)] for i in range(5)]
        kl=sum(sum(p*math.log(max(p,1e-12)/max(q,1e-12)) for p,q in zip(row['probabilities'],probabilities(proposed,row['features']))) for row in rows)/len(rows)
        if math.isfinite(kl) and kl<=.01:break
        step*=.5
    else:raise ValueError('Unstable online update')
    learned=copy.deepcopy(model)
    learned.update(weights=proposed,value_weights=[max(-20,min(20,v+.05*g)) for v,g in zip(value,value_gradient)],updates=model.get('updates',0)+1)
    stats=dict(transitions=len(rows),reward_sum=sum(r['reward'] for r in rows),gradient_norm=norm,
               weight_change_l2=step*scale*norm,kl=kl,step_size=step,return_mean=sum(returns)/len(rows))
    return learned,stats
