// turn below off for final checking
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]
//! # Some code to demonstrate the use of the ising model
//!
use hello_rust::*;
use std::cell::RefCell;
use std::os::windows::thread;
use std::vec;

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

//TODO: sort out rows and columns for consistency; not sure at all if I'm using the terms consistently rn
// also, n_cols vs n_columns

/// Main function that runs the Ising model simulation.
/// It creates a lattice, and then runs the simulation for a number of iterations.
/// For demonstration purposes it changes the temperature of the lattice over time.
/// It then plots the magnetization and energy of the lattice over time.
/// you can see the curie temperature around T ~ 2, even without an external magnetic field!
/// Check out the energy vs T plot to see this.
/// The "hamiltonian" function can be (ab)used to create lattices that simulate
///  something other than the ising model.  `conway_life_hamiltonian` is an example
/// that simulates Conway's Game of Life.
fn main() {
    let c = load_config(None);

    let hamiltonian = standard_ising_hamiltonian();
    let mut lattice = Lattice::new(&c, hamiltonian);

    //let mut lattice = Lattice::new(&c, conway_life_hamiltonian);

    let ffs = lattice.n_columns / 2;

    let max_iter = c.max_iter;
    let mut mag_over_time: Vec<f64> = Vec::new();
    let mut energy_over_time: Vec<f64> = Vec::new();
    let mut temperature_over_time: Vec<f64> = Vec::new();
    let approx_n_per_thread = lattice.n_columns * lattice.n_rows / lattice.thread_pool.len();
    println!("Approx n per thread: {}", approx_n_per_thread);

    for idx_t in 0..max_iter {
        let mut current_temp = 4.01 - (idx_t as f64 / (max_iter as f64) * 5.0);
        if current_temp < 0.01 {
            current_temp = 0.01;
        }
        lattice.set_temperature(current_temp);
        let current_mag = lattice.net_magnetization();
        mag_over_time.push(current_mag);
        energy_over_time.push(lattice.energy());
        temperature_over_time.push(current_temp);
        // if idx_t % (max_iter / 10) == 0 || idx_t == max_iter - 1 {
        //     let real_temperature = lattice.temperature.read().unwrap()[(0, 0)]; //get the _actual_ temp as the lattice sees it.
        //     println!("Temperature: {}", real_temperature);
        //     lattice.moments_as_heatmap(
        //         format!("{idx_t}.png"),
        //         format!("{}", real_temperature),
        //         false,
        //     );
        // }

        lattice.update_n_per_thread_random(approx_n_per_thread / 5);
    }
    lattice.shutdown_threads();

    //now plot the magnetization and energy over time in a web browser
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
