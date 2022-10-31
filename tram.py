# street with blocks numbered from 1 to n
# walking from s to s+1 takes 1 min
# taking a magic tram from s to 2s takes 2 mins
# tram fails with some probability . If it fails, we don't move but we also lose time.
# how to travel from 1 to n in the least time?
from enum import Enum
from faulthandler import is_enabled
import numpy as np

class Action(Enum):
    NONE = 0
    WALK = 1
    TRAM = 2

    @classmethod
    def default_action(cls):
        return Action.WALK

class TransportationMDP():

    def __init__(self, N:int, start_state:int=1, discount:float=1.0) -> None:
        self.N:int = N # uint to be precise
        self.start:int = start_state # uint to be precise
        self.gamma:float = discount

    def get_states(self) -> list[int]:
        return list(range(1, self.N + 1))

    def start_state(self) -> int:
        return self.start
    
    def is_end(self, state:int) -> bool:
        return state == self.N
    
    def actions_at_state(self, state:int) -> list[Action]:
        result = []
        if state + 1 <= self.N:
            result.append(Action.WALK)
        if state * 2 <= self.N:
            result.append(Action.TRAM)
        return result

    def prob_reward(self, state:int, action:Action) -> list[tuple[int, float, float]]:
        '''
            state = s, action = a, new_state = s'
            prob = T(s, a, s'), reward = r(s, a, s')
            returns list of (new_state, prob, reward)
        '''
        result = []

        match action:
            case Action.WALK:
                result.append((state + 1, 1.0, -1.))
            case Action.TRAM:
                result.append((state * 2, 0.5, -2.))
                result.append((state, 0.5, -2.))

        return result

def value_iteration(mdp:TransportationMDP, epsilon:float=1e-5):
    v = np.zeros(mdp.N)
    all_states = mdp.get_states()
    offset = mdp.start_state()
    def Q(state:int, action:Action) -> float:
        # computing q
        return sum(
            p * (r + mdp.gamma * v[st-offset]) \
            for st, p, r in mdp.prob_reward(state, action)
        )
    
    while True:
        vs = np.zeros(mdp.N)
        # Updating value of state
        for s in all_states:
            # updating value of state and policy
            if not mdp.is_end(s):
                vs[s - offset] = max(Q(s, a) for a in mdp.actions_at_state(s)) # using current V
        # check convergence
        if np.max(np.abs(v-vs)) < epsilon:
            break 
        v = vs
    
    # return policy, computed using the best V
    return [Action.NONE if mdp.is_end(s) else max(((Q(s, a), a) for a in mdp.actions_at_state(s)), key = lambda x:x[0])[1]\
             for s in all_states]

def policy_iteration(mdp:TransportationMDP, epsilon:float=1e-5):
    offset = mdp.start_state()
    all_states = mdp.get_states()
    v = np.zeros(mdp.N)
    pi = [Action.default_action()] * mdp.N # 'random' policy
    def Q(state:int, action:Action) -> float:
        # computing q
        return sum(
            p * (r + mdp.gamma * v[st- offset]) \
            for st, p, r in mdp.prob_reward(state, action)
        )

    while True:
        while True:
            old_v = v.copy()
            for i,s in enumerate(all_states):
                if not mdp.is_end(s):
                    v[i] = Q(s, pi[i])

            if np.max(np.absolute(v-old_v)) < epsilon:
                break

        stable = True
        for i,s in enumerate(all_states):
            if not mdp.is_end(s):
                old_action = pi[i]
                pi[i] = max(((Q(s, a), a) for a in mdp.actions_at_state(s)), key = lambda x:x[0])[1]
                if old_action != pi[i]:
                    stable = False
        if stable:
            break

    return pi

# mdp = TransportationMDP(10)
# print(policy_iteration(mdp))