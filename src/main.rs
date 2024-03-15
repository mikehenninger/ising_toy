// turn below off for final checking before "a release"
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
//! # doc test
//!
use hello_rust::*;

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
    let mut lattice = Lattice::new(N_ROWS, N_COLUMNS, a_hamiltonian, hmap);

    let new_vec_external_field = (-250..250)
        .map(|x| (x as f64) * 0.0)
        .collect::<Vec<f64>>()
        .repeat(lattice.n_rows);
    let mut new_temperature = Matrix::new(
        lattice.n_rows,
        lattice.n_columns,
        vec![0.0; lattice.n_rows * lattice.n_columns],
    );

    // for i in 0..lattice.n_rows {
    //     for j in 0..lattice.n_columns {
    //         //i+1 to avoid div by zero when calculating boltzmann factor
    //         new_temperature[(i, j)] = ((i + 1) as f64) / 50.0;
    //     }
    // }
    let mut new_ext_field = Matrix::new(lattice.n_rows, lattice.n_columns, new_vec_external_field);
    //new_ext_field[(0, 0)] = 50.0;
    let mut plot = Plot::new();
    let trace = HeatMap::new_z(new_ext_field.as_vec_vec());
    let title_ext_field = Title::new("External Field");
    let layout = Layout::new().title(title_ext_field).width(800).height(800);
    let layout2 = layout.clone().title(Title::new("Temperature"));

    plot.add_trace(trace);
    plot.set_layout(layout);

    plot.show();
    let mut plot = Plot::new();
    let trace = HeatMap::new_z(new_temperature.as_vec_vec());

    plot.add_trace(trace);
    plot.set_layout(layout2);
    plot.show();

    //lattice.sequential_update();
    lattice.set_external_field(new_ext_field.clone());
    //lattice.set_temperature(new_temperature);
    let max_iter = 2000;
    let mut mag_over_time: Vec<f64> = Vec::new();
    let mut energy_over_time: Vec<f64> = Vec::new();
    let mut temperature_over_time: Vec<f64> = Vec::new();
    for idx_t in 0..max_iter {
        let mut current_temp = 5.01 - (idx_t as f64 / (max_iter as f64) * 5.0);
        lattice.set_temperature(current_temp);

        lattice.sequential_update();
        let current_energy = lattice.energy();
        println!("Energy: {}", current_energy);
        energy_over_time.push(current_energy);
        let current_mag = lattice.net_magnetization();
        println!("Magnetization: {}", current_mag);
        mag_over_time.push(current_mag);

        temperature_over_time.push(current_temp);
        if idx_t % (max_iter as f64 / 1.01) as i32 == 0 {
            lattice.moments_as_heatmap(format!(" current_temp: {}", current_temp));
        }

        println!("Temperature: {}", current_temp);
    }
    println!("Energy: {}", lattice.energy());
    let mut mag_plot = Plot::new();
    let mag_trace = plotly::Scatter::new(temperature_over_time.clone(), mag_over_time);
    let layout = Layout::new().title(Title::new("Magnetization vs temperature"));
    mag_plot.add_trace(mag_trace);
    mag_plot.set_layout(layout);
    mag_plot.show();

    let mut energy_plot = Plot::new();
    let energy_trace = plotly::Scatter::new(temperature_over_time, energy_over_time);
    let layout = Layout::new().title(Title::new("Energy vs temperature"));
    energy_plot.add_trace(energy_trace);
    energy_plot.set_layout(layout);
    energy_plot.show();
}
