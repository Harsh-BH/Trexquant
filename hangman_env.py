import random
import re

class HangmanEnv:
    """
    A very simple Hangman environment:
      - State is (pattern, guessed_letters)
      - Actions are letters 'a'..'z'
      - Reward: +1 if correct guess, -1 if incorrect, 0 on non‐terminal step
      - Episode ends when word is fully revealed or max_incorrect reached.
    """
    def __init__(self, word_file="words_250000_train.txt", max_incorrect=6):
        # load words once
        with open(word_file, 'r') as f:
            # keep only lowercase alpha words
            self.word_list = [w.strip().lower() for w in f
                              if w.strip().isalpha()]
        self.max_incorrect = max_incorrect
        self.reset()

    def reset(self):
        self.target = random.choice(self.word_list)
        self.pattern = ['_'] * len(self.target)
        self.guessed = set()
        self.incorrect = 0
        return self._get_obs()

    def _get_obs(self):
        # returns (string pattern, sorted guessed letters list)
        return ''.join(self.pattern), sorted(self.guessed)

    def step(self, action):
        """
        action: single-letter string
        returns: next_obs, reward, done, info
        """
        reward = 0
        done = False
        letter = action.lower()
        if letter in self.guessed or not re.fullmatch(r'[a-z]', letter):
            # illegal or repeated guess: small penalty
            reward = -0.5
        else:
            self.guessed.add(letter)
            if letter in self.target:
                for i, c in enumerate(self.target):
                    if c == letter:
                        self.pattern[i] = letter
                reward = +1.0
            else:
                self.incorrect += 1
                reward = -1.0

        if '_' not in self.pattern:
            done = True
            reward += 5.0   # bonus for finishing word
        elif self.incorrect >= self.max_incorrect:
            done = True
            reward -= 5.0  # penalty for failure

        return self._get_obs(), reward, done, {}