#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hello_rust::*;
fn to_be_benched() {
    let c = load_config();
    let hsize: (usize, usize) = (3, 3);
    let mut hmap = Matrix::new(hsize.0, hsize.1, vec![1.0; hsize.0 * hsize.1]);
    hmap[(0, 0)] = 0.0;
    hmap[(0, 2)] = 0.0;
    hmap[(2, 0)] = 0.0;
    hmap[(2, 2)] = 0.0;
    hmap[(1, 1)] = 0.0;
    let hamiltonian = mapped_hamiltonian(&hmap);
    let mut lattice = AlternateLattice::new(&c, hamiltonian);

    let new_vec_temps = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.].repeat(lattice.n_rows);

    for idx_t in 0..100 {
        lattice.full_update();
        //lattice.update((1, 1))
    }
}
fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("ising", |b| b.iter(|| to_be_benched()));
}
criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark

}
//criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
