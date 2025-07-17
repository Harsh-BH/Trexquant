import os
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from hangman_env import HangmanEnv
from vocab import LETTERS, LETTER2IDX, UNK_IDX, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM

# -------------------------
# 1. Actor‐Critic Network
# -------------------------
class ActorCriticNet(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # rows = VOCAB_SIZE
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.policy_fc = nn.Linear(hidden_dim, vocab_size)
        self.value_fc  = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        h = out[:, -1, :]
        logits = self.policy_fc(h)
        value  = self.value_fc(h).squeeze(-1)
        return logits, value, hidden

# -------------------------
# 2. State encoding
# -------------------------
def encode_state(pattern: str, device) -> torch.Tensor:
    seq = [LETTER2IDX.get(ch, UNK_IDX) for ch in pattern]
    return torch.tensor([seq], dtype=torch.long, device=device)

# -------------------------
# 3. Letter‐frequency prior
# -------------------------
def letter_frequency_prior(candidates, device):
    cnt = {}
    for w in candidates:
        for ch in set(w):
            cnt[ch] = cnt.get(ch, 0) + 1
    freq = torch.zeros(VOCAB_SIZE - 1, device=device)  # only real letters
    for ch, c in cnt.items():
        idx = LETTER2IDX.get(ch, None)
        if idx is not None:
            freq[idx] = c
    if freq.sum() > 0:
        freq = freq / freq.sum()
    # pad for the unk index
    return torch.cat([freq, torch.tensor([0.0], device=device)], dim=0)

# -------------------------
# 4. Training Loop (A2C)
# -------------------------
def train(
    word_file="words_250000_train.txt",
    max_incorrect=6,
    n_episodes=30000,
    gamma=0.99,
    lr=3e-4,
    value_coef=0.5,
    entropy_coef=0.01,
    save_path="models/policy.pt",
    device="cpu"
):
    env = HangmanEnv(word_file=word_file, max_incorrect=max_incorrect)
    net = ActorCriticNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    reward_hist = deque(maxlen=200)
    for ep in range(1, n_episodes+1):
        state, guessed = env.reset()
        done = False
        ep_reward = 0.0
        candidates = env.word_list.copy()
        hidden = None

        log_probs, values, rewards, entropies, masks = [], [], [], [], []

        while not done:
            regex = "^" + state.replace("_", ".") + "$"
            candidates = [
                w for w in candidates
                if len(w)==len(state) and re.fullmatch(regex, w)
            ]

            state_t = encode_state(state, device)
            logits, value, hidden = net(state_t, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            probs = F.softmax(logits, dim=-1).squeeze(0)
            mask = torch.ones_like(probs)
            for g in guessed:
                if g in LETTER2IDX:
                    mask[LETTER2IDX[g]] = 0.0
            probs = probs * mask
            if probs.sum().item() == 0:
                probs = torch.ones_like(probs)
            probs = probs / probs.sum()

            alpha = 0.1
            prior = letter_frequency_prior(candidates, device)
            probs = (1 - alpha)*probs + alpha*prior
            probs = probs / probs.sum()

            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            log_probs.append(dist.log_prob(idx))
            entropies.append(dist.entropy())
            values.append(value.squeeze(0))

            action = LETTERS[idx.item()]
            (state, guessed), r, done, _ = env.step(action)
            rewards.append(r)
            masks.append(1 - int(done))
            ep_reward += r

        # compute GAE returns & advantages
        returns, advantages = [], []
        R = A = last_val = 0
        lam = 0.95
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma*R*masks[i]
            td = rewards[i] + gamma*last_val*masks[i] - values[i]
            A = td + gamma*lam*A*masks[i]
            last_val = values[i]
            returns.insert(0, R)
            advantages.insert(0, A)

        returns = torch.tensor(returns, device=device)
        advantages = torch.tensor(advantages, device=device)
        if advantages.std().item() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        policy_loss  = -(torch.stack(log_probs) * advantages).sum()
        value_loss   = value_coef * F.mse_loss(torch.stack(values), returns)
        entropy_loss = -entropy_coef * torch.stack(entropies).sum()
        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reward_hist.append(ep_reward)
        if ep % 500 == 0:
            avg_r = sum(reward_hist) / len(reward_hist)
            print(f"Ep {ep}/{n_episodes} | avgR {avg_r:.2f}")

        if len(reward_hist)==reward_hist.maxlen and (sum(reward_hist)/len(reward_hist) > max_incorrect*0.8):
            print(f"Solved by ep {ep}! avg reward {sum(reward_hist)/len(reward_hist):.2f}")
            break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"model_state": net.state_dict()}, save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()