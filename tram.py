# street with blocks numbered from 1 to n
# walking from s to s+1 takes 1 min
# taking a magic tram from s to 2s takes 2 mins
# tram fails with prob 0.01. If it fails, we don't move but we also lose time.
# how to travel from 1 to n in the least time?
from enum import Enum
import numpy as np 

class Action(Enum):
    WALK = 1
    TRAM = 2

class TransportationMDP():

    def __init__(self, N:int) -> None:
        self.N = N

    def list_states(self) -> None:
        print(list(range(1,self.N+1)))

    def start_state(self) -> int:
        return 1
    
    def is_end(self, state:int) -> bool:
        return state == self.N
    
    # look later
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
                result.append((state * 2, 0.9, -2.))
                result.append((state, 0.1, -2.))

        return result

def value_iteration(mdp:TransportationMDP):
    v = {} 


mdp = TransportationMDP(10)
print(mdp.prob_reward(3, Action.TRAM))
