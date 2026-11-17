"""Vanilla policy gradient with a linear value baseline and a KL-limited step.

No GPU dependency: this high-level policy has only 60 actor parameters. Physics,
not the gradient calculation, dominates training time.
"""
import copy
import numpy as np


def softmax(logits):
    e=np.exp(logits-logits.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def policy_gradient(weights,x,actions,advantages,entropy=.01):
    probs=softmax(x@weights.T)
    onehot=np.eye(weights.shape[0])[actions]
    grad=((onehot-probs)*advantages[:,None]).T@x/len(x)
    h=-(probs*np.log(np.maximum(probs,1e-12))).sum(axis=1)
    entropy_grad=(-probs*(np.log(np.maximum(probs,1e-12))+h[:,None])).T@x/len(x)
    return grad+entropy*entropy_grad


def update(model,rows,gamma=.99,learning_rate=2.0):
    if len(rows)<20:raise ValueError('Too few valid transitions for an RL update')
    x=np.asarray([r['features'] for r in rows],dtype=np.float64)
    actions=np.asarray([r['action'] for r in rows],dtype=int)
    weights=np.asarray(model['weights'],dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=12 or weights.shape!=(5,12):raise ValueError('Policy shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(weights).all():raise ValueError('Non-finite training input')
    old=softmax(x@weights.T)
    recorded=np.asarray([r['probabilities'] for r in rows])
    if not np.allclose(old,recorded,atol=1e-7,rtol=1e-7):
        raise ValueError('Rollout probabilities differ from checkpoint: refusing off-policy update')
    returns=[]
    future=0.
    for row in reversed(rows):
        discount=gamma**max(.01,row.get('duration_sim_seconds',2)/2)
        future=float(row['reward'])+discount*(0 if row.get('done') else future)
        returns.append(future)
    returns=np.asarray(list(reversed(returns)))
    baseline=x@np.asarray(model.get('value_weights',[0.]*12))
    advantage=returns-baseline
    advantage=(advantage-advantage.mean())/max(float(advantage.std()),1e-6)
    grad=policy_gradient(weights,x,actions,advantage)
    norm=float(np.linalg.norm(grad))
    grad*=min(1.,5/max(norm,1e-12))
    step=learning_rate
    for _ in range(20):
        proposed=weights+step*grad
        new=softmax(x@proposed.T)
        kl=float(np.mean(np.sum(old*np.log(np.maximum(old,1e-12)/np.maximum(new,1e-12)),axis=1)))
        if np.isfinite(proposed).all() and kl<=.02:break
        step*=.5
    else:raise ValueError('Could not obtain a finite bounded policy update')
    value=np.linalg.solve(x.T@x+.1*np.eye(12),x.T@returns)
    result=copy.deepcopy(model)
    result['weights']=proposed.tolist()
    result['value_weights']=value.tolist()
    result['updates']=model.get('updates',0)+1
    stats=dict(transitions=len(rows),reward_sum=float(sum(r['reward'] for r in rows)),
               gradient_norm=norm,weight_change_l2=float(np.linalg.norm(proposed-weights)),
               kl=kl,step_size=step,return_mean=float(returns.mean()),
               action_counts=np.bincount(actions,minlength=5).tolist())
    return result,stats
