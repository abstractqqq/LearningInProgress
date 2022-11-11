#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    NONE,
    WALK,
    TRAM
}

impl Action {
    fn default() -> Action {
        Action::WALK
    }
}

// pub trait MarkovDecisionProcess {
//     // For now, just use usize to represent states. Can be generalized later.
//     fn get_start_state(&self) -> usize;
//     // fn get_action_at_state(&self, state:usize) -> Vec<Action>;
//     fn get_discount(&self) -> f32;
//     // fn get_states_as_iter(&self) -> dyn Iterator<Item = usize>;
// }

// pub trait StateAction{
//     fn get_action(&self, T:_) -> Action;
// }

pub struct Transportation {
    start_state : usize,
    n: usize,
    gamma : f32
}

// impl MarkovDecisionProcess for Transportation {
//     fn get_start_state(&self) -> usize {
//         self.start_state
//     }

//     fn get_action_at_state(&self, state:usize) -> Vec<Action>{

//         let mut actions:Vec<Action> = Vec::new();
//         if state + 1 <= self.n {
//             actions.push(Action::WALK);
//         }
//         if state * 2 <= self.n {
//             actions.push(Action::TRAM);
//         }

//         actions 

//     }

//     fn get_discount(&self) -> f32 {
//         self.gamma
//     }

// }

impl Transportation {
    pub fn new(start:usize, total_states:usize, discount:f32) -> Transportation{
        Transportation {start_state:start, n: total_states, gamma:discount}
    }

    fn get_start_state(&self) -> usize {
        self.start_state
    }


    fn get_all_states(&self) -> Vec<usize>{
        (1..=self.n).collect()
    }

    fn get_discount(&self) -> f32 {
        self.gamma
    }

    fn get_action_at_state(&self, state:usize) -> Vec<Action>{

        let mut actions:Vec<Action> = Vec::new();
        if state + 1 <= self.n {
            actions.push(Action::WALK);
        }
        if state * 2 <= self.n {
            actions.push(Action::TRAM);
        }

        actions 

    }

    fn get_prob_reward(&self, state:usize, action:&Action) -> Vec<(usize, f32, f32)> {
        // return probabilities of possible actions and the possible next states of these actions
        let mut output:Vec<(usize, f32, f32)> = Vec::new();
        match action {
            Action::WALK => output.push((state + 1, 1.0, -1.)),
            Action::TRAM => {
                output.push((state * 2, 0.5, -2.));
                output.push((state, 0.5, -2.));
            }
            _ => {}
        }
        output 

    }

    fn is_end(&self, state:usize) -> bool {
        self.n == state 
    }
}


pub fn policy_iteration(mdp:&Transportation, epislon:f32) -> Vec<Action>{

    fn q(mdp:&Transportation, cur_v:&Vec<f32>, state:usize, a:&Action) -> f32 {
        let q_states = mdp.get_prob_reward(state, a);
        q_states.into_iter().fold(0., 
            |acc:f32, (st, p, r)| acc + p * (r + mdp.get_discount() * cur_v[st-mdp.get_start_state()])
        )
    }

    let mut pi:Vec<Action> = vec![Action::default(); mdp.n];
    let mut v:Vec<f32> = vec![0.;mdp.n];
    
    loop {
        loop { // value of a policy
            let mut max_diff:f32 = 0.;
            for (i, s) in (mdp.get_start_state()..=mdp.n).enumerate() {
                if !mdp.is_end(s){
                    let old_v = v[i]; 
                    v[i] = q(&mdp, &v, s, &pi[i]);
                    let diff = (old_v - v[i]).abs();
                    if diff > max_diff {max_diff = diff;}
                }
                // element-wise check max |difference|, which is equivalent to L1 between two vectors.
                // This approach might be faster because I am not copying vectors and doing non-optimized vector L1 norm computations.
                // Purely scalar computation. 
            }
            // check convergence, e.g. L1(V - OLD_V) < epsilon
            if max_diff < epislon {
                break 
            }
        }
        let mut stable = true;
        for (i, s) in (mdp.get_start_state()..=mdp.n).enumerate() {
            if mdp.is_end(s){
                pi[i] = Action::NONE;
            } else {
                let old_action = pi[i].clone(); // Probably cheap.. just copying an enum..
                let all_possible_actions = mdp.get_action_at_state(s);
                let action_values = all_possible_actions.into_iter().map(
                    |a| (q(&mdp, &v, s, &a), a)
                );
                let new_action = action_values.reduce(|accum, value_action_pair| {
                    if value_action_pair.0 > accum.0 {value_action_pair} else {accum}});
                match new_action {
                    Some(value_pair) => {
                        pi[i] = value_pair.1;
                        if pi[i] != old_action {stable = false;}
                    },
                    _ => {}
                }
            }
        }
        if stable {break}
    }
    pi
}



