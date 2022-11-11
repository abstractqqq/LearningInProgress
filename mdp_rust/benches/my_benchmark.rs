use criterion::{criterion_group, criterion_main, Criterion};
use mdp_rust::mdp::{Action, Transportation, policy_iteration};


fn poli_iter_test() -> Vec<Action> {
    let game = Transportation::new(1, 100, 1.0);
    policy_iteration(&game, 1e-7)
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("poli_iter_test", |b| b.iter(|| poli_iter_test()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);