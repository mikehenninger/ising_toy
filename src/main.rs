// turn below off for final checking
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
//! # doc test
//!
use hello_rust::*;
use std::cell::RefCell;

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

//TODO: sort out rows and columns for consistency; not sure at all if I'm using the terms consistently rn
// also, n_cols vs n_columns

fn main() {
    let hsize: (usize, usize) = (3, 3);
    let mut hmap = Matrix::new(hsize.0, hsize.1, vec![1.0; hsize.0 * hsize.1]);
    hmap[(0, 0)] = 0.0;
    hmap[(0, 2)] = 0.0;
    hmap[(2, 0)] = 0.0;
    hmap[(2, 2)] = 0.0;
    hmap[(1, 1)] = 0.0;

    let hamiltonian = mapped_hamiltonian(&hmap);
    let mut lattice = AlternateLattice::new(N_ROWS, N_COLUMNS, hamiltonian);
    let ffs = lattice.n_columns / 2;
    let new_vec_external_field = (-(ffs as i64)..(ffs as i64))
        .map(|x| (x as f64) * 0.0)
        .collect::<Vec<f64>>()
        .repeat(lattice.n_rows);
    let mut new_temperature = Matrix::new(
        lattice.n_rows,
        lattice.n_columns,
        vec![0.0; lattice.n_rows * lattice.n_columns],
    );
    let mut new_ext_field = Matrix::new(lattice.n_rows, lattice.n_columns, new_vec_external_field);

    //lattice.sequential_update();
    lattice.set_external_field(new_ext_field.clone());
    lattice.set_temperature(new_temperature);
    let max_iter = 10;
    let mut mag_over_time: Vec<f64> = Vec::new();
    let mut temperature_over_time: Vec<f64> = Vec::new();
    lattice.sequential_update();
    for idx_t in 0..max_iter {
        let mut current_temp = 5.01 - (idx_t as f64 / (max_iter as f64) * 5.0);
        //let current_temp = 0.5;
        lattice.set_temperature(current_temp);
        //println!("Energy: {}", lattice.energy());
        let current_mag = lattice.net_magnetization();
        //println!("Magnetization: {}", current_mag);
        mag_over_time.push(current_mag);

        temperature_over_time.push(current_temp);
        if idx_t % (max_iter / 10) as i32 == 0 {
            lattice.moments_as_heatmap(format!("{idx_t}.png"), false);
            println!("Temperature: {}", current_temp);
        }
        lattice.sequential_update();

        //println!("Temperature: {}", current_temp);
    }
    // println!("Energy: {}", lattice.energy());
    // let mut mag_plot = Plot::new();
    // let mag_trace = plotly::Scatter::new(temperature_over_time, mag_over_time);
    // let layout = Layout::new().title(Title::new("Magnetization vs temperature"));
    // mag_plot.add_trace(mag_trace);
    // mag_plot.set_layout(layout);
    // mag_plot.show();
}
