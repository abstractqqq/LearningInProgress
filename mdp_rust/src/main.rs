mod mdp;
use mdp::{Transportation, policy_iteration};

fn main() {
    let game = Transportation::new(1, 20, 1.0);
    let optimal_policy = policy_iteration(&game, 1e-7);
    println!("{:?}", optimal_policy);

}