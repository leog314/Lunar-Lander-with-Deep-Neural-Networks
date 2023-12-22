import gym
import random
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
import matplotlib.pyplot as plt
import time

fig, [ax, ax1] = plt.subplots(2, 1)
# create plot

data = []
y = []
losses = []

batch_size = 4
all_steps = 0
epochs = 2000
# Hyperparameters for training

env = gym.make(
    "LunarLander-v2",
    render_mode="human",
)

class Memory:
    def __init__(self, cap):
        self.cap = cap
        self.mem = []
        self.pos = 0

    def push(self, state_, action_, nstate_, reward, done_):
        self.mem.append((state_, action_, nstate_,reward, done_))
        self.mem.pop(0) if len(self.mem) > self.cap else None
        # pushes data into memory

    def sample(self, batch_size_):
        return random.sample(self.mem, batch_size_) # returns a random sample

    def __len__(self):
        return len(self.mem)


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.logits = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
        )
        # the Agents neural network

        self.crit = nn.SmoothL1Loss() # loss function, either MSELoss or even better: SmoothL1Loss
        self.optim = optimizer.Adam(self.parameters(), lr=0.003) 
        self.schedule = optimizer.lr_scheduler.PolynomialLR(self.optim, total_iters=10**6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


mem = Memory(2048)
model = Agent()
gamma = 0.95 # gamma > 0.9

eps_start = 0.99
eps_end = 0.05
steps = 10000
d = 0
t = 1

def get_action(in_state: torch.tensor) -> int:
    model.eval()
    global d
    global t
    eps = random.random()
    t = eps_end + (eps_start-eps_end)*np.exp(-d/steps)
    d += 1
    if eps > t:
        with torch.no_grad():
            return int(torch.argmax(model(in_state)))
    else:
        return random.randint(0, 3)

def training() -> float:
    model.train()
    model.optim.zero_grad(True)

    state_ten, action_ten, nstate_ten, rew_ten, done_ten = zip(*mem.sample(batch_size))

    state_ten = torch.stack(state_ten)
    action_ten = torch.stack(action_ten)
    nstate_ten = torch.stack(nstate_ten)
    rew_ten = torch.stack(rew_ten)
    done_ten = torch.stack(done_ten)

    out = model(state_ten)
    target = out.clone()

    for i in range(batch_size):
        target[i, int(action_ten[i])] = rew_ten[i] + gamma*torch.max(model(nstate_ten[i]), dim=0).values.unsqueeze(0) * (1-done_ten[i].long())

    loss = model.crit(out, target)
    loss.backward()

    model.optim.step()
    model.schedule.step()

    return loss.item()

for epoch in range(epochs):
    ste = 0
    rews = 0
    los = 0
    start_time = time.perf_counter()

    if epoch > 50:
        data = data[1:]
        y = y[1:]
        losses = losses[1:]

    obs = env.reset()
    state = torch.Tensor(obs[0])
    done = False
    env.render()

    while not done:
        action = get_action(state)
        nobs, rew, done, _, _ = env.step(action)
        rew = torch.Tensor([rew])

        if time.perf_counter()-start_time > 45:
            break

        nstate = torch.from_numpy(nobs)

        mem.push(state, torch.Tensor([action]), nstate, rew, torch.Tensor([done]))

        state = torch.clone(nstate)
        error = training() if len(mem) > batch_size else 0
        los += error

        rews += float(rew)
        ste += 1
        all_steps += 1

    data.append(epoch)
    y.append(rews/ste)
    losses.append(los/ste)

    ax.clear()
    ax1.clear()

    ax.plot(data, y, "bo", linestyle="dashed")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"{rew} {all_steps}")

    ax1.plot(data, losses, linestyle="dashed", color="green")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Average loss")
    plt.pause(0.01)

    print(f"Average reward: {rews/ste} - Average loss: {los/ste} - {epoch} ({int(rew)})")
    print(t, model.schedule.get_last_lr())

plt.show()
# torch.save(model, "rl_model_f2.pt")
env.close()
