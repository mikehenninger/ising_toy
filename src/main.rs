//! # doc test
use hello_rust::*;

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

fn main() {
    let hsize: (usize, usize) = (3, 3);
    let mut hmap = Matrix::new(hsize.0, hsize.1, vec![-1.0; hsize.0 * hsize.1]);
    hmap[(0, 0)] = 0.0;
    hmap[(0, 2)] = 0.0;
    hmap[(2, 0)] = 0.0;
    hmap[(2, 2)] = 0.0;
    hmap[(1, 1)] = 0.0;
    let mut lattice = Lattice::new(N_ROWS, N_COLUMNS, a_hamiltonian, hmap);

    let new_vec_temps =
        [-20., 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1.5, 20.].repeat(lattice.n_columns);
    println!("{:}", new_vec_temps.len());
    let new_temps = Matrix::new(lattice.n_rows, lattice.n_columns, new_vec_temps);

    let mut plot = Plot::new();
    let trace = HeatMap::new_z(new_temps.as_vec_vec());
    plot.add_trace(trace);
    plot.show();
    //lattice.sequential_update();
    lattice.set_external_field(new_temps);
    for _ in 0..10 {
        println!("Energy: {}", lattice.energy());
        lattice.sequential_update();
        lattice.moments_as_heatmap();
    }
    println!("Energy: {}", lattice.energy());
}
