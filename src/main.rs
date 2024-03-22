// turn below off for final checking
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
//! # doc test
//!
use hello_rust::*;
use std::cell::RefCell;
use std::os::windows::thread;
use std::vec;

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

//TODO: sort out rows and columns for consistency; not sure at all if I'm using the terms consistently rn
// also, n_cols vs n_columns

fn main() {
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
    // let ffs = lattice.n_columns / 2;
    // let new_vec_external_field = (-(ffs as i64)..(ffs as i64))
    //     .map(|x| (x as f64) * 0.05)
    //     .collect::<Vec<f64>>()
    //     .repeat(lattice.n_rows);
    //lattice.sequential_update();
    let max_iter = c.max_iter;
    let mut mag_over_time: Vec<f64> = Vec::new();
    let mut energy_over_time: Vec<f64> = Vec::new();
    let mut temperature_over_time: Vec<f64> = Vec::new();
    let approx_n_per_thread = lattice.n_columns * lattice.n_rows / lattice.thread_pool.len();
    println!("Approx n per thread: {}", approx_n_per_thread);
    lattice.full_update(); //XXX needed to fill scratch, but why is that required?
    for idx_t in 0..max_iter {
        let mut current_temp = 4.01 - (idx_t as f64 / (max_iter as f64) * 5.0);
        if current_temp < 0.01 {
            current_temp = 0.01;
        }
        lattice.set_temperature(current_temp);
        //println!("Energy: {}", lattice.energy());
        let current_mag = lattice.net_magnetization();
        // //println!("Magnetization: {}", current_mag);
        mag_over_time.push(current_mag);
        energy_over_time.push(lattice.energy());
        temperature_over_time.push(current_temp);
        if idx_t % (max_iter / 11) == 0 {
            lattice.moments_as_heatmap(format!("{idx_t}.png"), false);
            println!(
                "Temperature: {}",
                lattice.temperature.read().unwrap()[(0, 0)]
            );
        }
        //lattice.update_one_per_thread_random();

        lattice.update_n_per_thread_random(approx_n_per_thread / 10);
    }
    //println!("Temperature: {}", current_temp);
    lattice.shutdown_threads();

    let mut mag_plot = Plot::new();
    let mag_trace = plotly::Scatter::new(temperature_over_time.clone(), mag_over_time);
    let layout = Layout::new().title(Title::new("Magnetization vs temperature"));
    mag_plot.add_trace(mag_trace);
    mag_plot.set_layout(layout);
    mag_plot.show();
    let mut energy_plot = Plot::new();
    let energy_trace = plotly::Scatter::new(temperature_over_time, energy_over_time);
    let layout = Layout::new().title(Title::new("energy vs temperature"));
    energy_plot.add_trace(energy_trace);
    energy_plot.set_layout(layout);
    energy_plot.show();
}
