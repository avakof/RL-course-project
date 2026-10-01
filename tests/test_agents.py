import ast,json,tempfile
import numpy as np
import torch
torch.set_num_threads(1)

def definitions(path):
 ns={'__name__':'verification'}
 for c in json.load(open(path))['cells']:
  if c['cell_type']!='code': continue
  tree=ast.parse(''.join(c['source']))
  keep=[]
  for node in tree.body:
   if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)):
    keep.append(node)
   elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id.endswith('Agent') for t in node.targets):
    keep.append(node)
  exec(compile(ast.Module(body=keep,type_ignores=[]),path,'exec'),ns)
 return ns
ns=definitions('TD1_Question.ipynb');a=ns['QLearningAgent'](lr=1)
s=np.zeros(4);idx=a.discretize_state(s)
assert a.discretize_state([-1e6]*4)==(0,0,0,0)
assert a.discretize_state([1e6]*4)==(9,9,9,9)
a.q_table.fill(10);a.update_q_table(s,0,1,s,True);assert a.q_table[idx+(0,)]==1
a.update_q_table(s,0,1,s,False);assert np.isclose(a.q_table[idx+(0,)],10.9)
ns=definitions('TD2_Question.ipynb');a=ns['SimpleDQNAgent']()
before=[p.clone() for p in a.q_network.parameters()]
assert np.isfinite(a.train_step(s,0,1,s,True))
assert any(not torch.equal(x,y) for x,y in zip(before,a.q_network.parameters()))
with tempfile.NamedTemporaryFile() as f:
 a.save_model(f.name);b=ns['SimpleDQNAgent']();b.load_model(f.name)
 assert all(torch.equal(x,y) for x,y in zip(a.q_network.parameters(),b.q_network.parameters()))
ns=definitions('TD3_Question.ipynb')
for cls in [ns['DQNAgent'],ns['DoubleDQNAgent']]:
 a=cls(batch_size=2,target_update_freq=2)
 assert a.train_step() is None
 assert all(x.data_ptr()!=y.data_ptr() and torch.equal(x,y) for x,y in zip(a.q_network.parameters(),a.target_q_network.parameters()))
 for _ in range(2):a.store_experience(s,0,1,s,True)
 original=[p.clone() for p in a.target_q_network.parameters()]
 assert np.isfinite(a.train_step())
 assert all(torch.equal(x,y) for x,y in zip(original,a.target_q_network.parameters()))
 a.train_step()
 assert all(torch.equal(x,y) for x,y in zip(a.q_network.parameters(),a.target_q_network.parameters()))
 assert all(p.grad is None for p in a.target_q_network.parameters())
print('PASS: discretization, terminal/bootstrap targets, gradient updates, checkpoint round trip, replay warm-up, independent target and synchronization for DQN/DDQN')

# Distinguish action selection from evaluation: online favors action 0,
# target favors action 1. Correct Double DQN bootstraps 2, ordinary DQN 9.
for cls, expected in [(ns['DQNAgent'],9.0),(ns['DoubleDQNAgent'],2.0)]:
    a=cls(batch_size=2,lr=0,gamma=1)
    with torch.no_grad():
        for p in a.q_network.parameters(): p.zero_()
        for p in a.target_q_network.parameters(): p.zero_()
        a.q_network.fc3.bias.copy_(torch.tensor([3.,1.]))
        a.target_q_network.fc3.bias.copy_(torch.tensor([2.,9.]))
    for _ in range(2): a.store_experience(s,0,0,s,False)
    assert np.isclose(a.train_step(), (3-expected)**2)
print('PASS: Double DQN selects online actions and evaluates them with the target network')
