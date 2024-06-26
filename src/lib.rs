// turn below off for final checking
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]

mod ultralight_matrix;
mod update_worker;

// all these imports are not idiomatic, but those types/identifiers/etc. are already so long
pub use ultralight_matrix::matrix::*;
pub use update_worker::*;

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};
use serde::{Deserialize, Serialize};

use std::sync::{mpsc, Arc, Barrier, Mutex, RwLock};

/// Represents the configuration for a simulation.
/// Typically loaded from disk using `load_config()`.

#[derive(Serialize, Deserialize)]
pub struct Config {
    // XXX TODO: max_iter is an odd one out, as it's not a `Lattice` parameter.
    // but I do want it to be changeable at runtime, so it's in config.json for now.
    n_rows: usize,    // number of rows of sites in the lattice
    n_columns: usize, // number of columns of sites in the lattice
    //below boils down to "these units are such that they plug directly into
    //making the boltzman factor exp(-external_field*moment/temperature), in the
    //absence of interaction terms"
    temperature: f64,     // in "natural units" propto KbT
    external_field: f64, // in natural units where 1.0 applies the same field at a site as its neighbors
    magnetic_moment: f64, // in natural units
    pub max_iter: usize, // number of iterations to run the simulation
    n_threads: usize,    // number of threads to use for parallel updates.
                         // on my machine, n_threads>1 only helps for n_sites>1e6 or so; less than that
                         // and the overhead + single-threaded parts dominate.
}

/// Loads the simulation configuration from a file.
/// If `filename` is provided, it reads the configuration from that file.
/// Otherwise, it reads from the default file "config.json".

pub fn load_config(filename: Option<&str>) -> Config {
    let config: String;
    match filename {
        Some(filename) => {
            config = std::fs::read_to_string(filename).unwrap();
        }
        None => {
            config = std::fs::read_to_string("config.json").unwrap();
        }
    }
    return serde_json::from_str(&config).unwrap();
}

/// The lattice struct is the main interface for the simulation.
/// It contains the lattice data, temperature, and external field as matrices
/// and it orchestrates each worker thread and its scratch region for making updates.
///
/// The lattice and its interactions wrap on both axes, so there are no edge effects...
/// at the cost of this actually being a simulation of a torroidal lattice, not a planar one.
/// Topologically speaking.
pub struct Lattice<F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send> {
    pub n_rows: usize,
    pub n_columns: usize,
    moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    scratch_moments: Vec<Arc<Mutex<Vec<f64>>>>,
    pub thread_pool: Vec<UpdateWorker<F>>,
    pub hamiltonian: F,
    scratch_barrier: Arc<Barrier>,
    can_wait_scratch_barrier: bool,
}

impl<F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send + 'static> Lattice<F> {
    /// Shuts down all worker threads in the lattice. Can't update after this is called.
    pub fn shutdown_threads(&mut self) -> () {
        for worker in &mut self.thread_pool {
            worker.sender.send(UpdateMessage::Stop).unwrap();

            if let Some(thread) = worker.thread.take() {
                thread.join().unwrap();
            }
        }
    }

    /// simple setter for temperature
    /// only twist is it needs to grab a write lock because it's shared with
    /// the worker threads
    pub fn set_temperature<T>(&mut self, temperature: T) -> ()
    where
        T: Broadcast,
    {
        let temp = temperature.broadcast((self.n_rows, self.n_columns));
        let mut lattice_temps = self.temperature.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                lattice_temps[(i, j)] = temp[(i, j)];
            }
        }
    }

    /// simple setter for external field
    /// only twist is it needs to grab a write lock because it's shared with
    /// the worker threads
    pub fn set_external_field<T>(&mut self, external_field: T) -> ()
    where
        T: Broadcast,
    {
        let field = external_field.broadcast((self.n_rows, self.n_columns));
        let mut lattice_external_field = self.external_field.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                lattice_external_field[(i, j)] = field[(i, j)];
            }
        }
    }

    /// returns a copy of the moments as a vector of vectors
    fn moments_as_vec_vec(&self) -> Vec<Vec<f64>> {
        let actual_moments = self.moments.read().unwrap();

        return actual_moments.copy_as_vec_vec();
    }
    // update now has a barrier that completes as first step of copy_from_scratch
    // tentatively think this is safe, as the barrier is only reached by worker thread
    // once per UpdateMessage sent.  Thus the `can_wait_scratch_barrier`
    // that `copy_from_scratch` checks and panics on, rather than leave you twisting
    pub fn full_update(&mut self) -> () {
        for worker in self.thread_pool.iter() {
            worker.sender.send(UpdateMessage::All).unwrap();
        }
        self.can_wait_scratch_barrier = true;
        self.copy_from_scratch();
    }

    /// tells workers to update a randomly site it owns and repeat that process
    /// (with replacement) n times.  This is dramatically faster than incremental
    /// update, but it's not a true random update like a single thread would be...
    /// all updates are against the same `moments` and don't reflect other nearby updates.
    /// And you can waste time updating the same site multiple times,
    /// though it shouldn't ruin your site outcome probabilities.
    pub fn update_n_per_thread_random(&mut self, n: usize) -> () {
        //if the last thread has fewer sites, those sites will be updated more frequently
        //so not ideal, statistically.

        for worker in self.thread_pool.iter() {
            worker.sender.send(UpdateMessage::UpdateN(n)).unwrap();
        }
        self.can_wait_scratch_barrier = true;
        self.copy_from_scratch();
    }
    /// send n_rows*n_columns UpdateMessages to the workers, one for each site
    /// brutally slow--much worse than single thread, but was useful for
    /// development purposes and is still illustrative.
    pub fn incremental_update(&mut self) -> () {
        let offset_list =
            (self.thread_pool.iter().map(|x| x.scratch_offset)).collect::<Vec<usize>>();
        let mut idx_linear: usize;
        let mut idx_thread: usize;
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                idx_linear = i * self.n_columns + j;

                idx_thread = find_slot_index(&offset_list, &idx_linear);
                //println!("Sending ({i}, {j}) to thread {idx_thread}");
                let update_loc = UpdateMessage::Location(i, j);
                self.thread_pool[idx_thread]
                    .sender
                    .send(update_loc)
                    .unwrap();
            }
        }
        self.can_wait_scratch_barrier = true;
        //copy_from_scratch has a barrier that ensures all threads have finished updating
        self.copy_from_scratch();
    }

    /// after all workers have updated their scratch regions, this copies the
    /// updated moments accumulated in `scratch_moments` back to `moments`
    /// this is a bottleneck in speed.  I need to see if vectorizing the copying
    /// helps, rather than looping over `scratch_loc`
    fn copy_from_scratch(&mut self) -> () {
        if !self.can_wait_scratch_barrier {
            panic!("Can't copy from scratch more than once per update cycle; will deadlock");
        }
        // wait here for all threads to be done writing scratch moments
        // can I remove this barrier if the thread locks its `scratch_moments`
        // mutex immediately on getting message?
        // I suspect that's still not safe
        self.scratch_barrier.wait();
        let mut actual_moments = self.moments.write().unwrap();
        for idx_thread in 0..self.thread_pool.len() {
            //println!("acquiring mutex for scratch moments from thread {idx_thread}");
            let scratch_moments = self.scratch_moments[idx_thread].lock().unwrap();
            //println!("got mutex for scratch moments from thread {idx_thread}");
            let scratch_offset = self.thread_pool[idx_thread].scratch_offset;
            for scratch_loc in 0..scratch_moments.len() {
                let absolute_loc = scratch_loc + scratch_offset;
                let (ui, uj) = (absolute_loc / self.n_columns, absolute_loc % self.n_columns);

                actual_moments[(ui, uj)] = scratch_moments[scratch_loc];
            }
        }
        self.can_wait_scratch_barrier = false;
    }

    /// once you've created or externally edited the moments, copy them to the scratch regions
    /// so all elements are present when you copy them back after an update--that
    /// might not have touched every scratch_site.
    fn copy_to_scratch(&mut self) -> () {
        let moments_read = self.moments.read().unwrap();
        for ui in 0..self.n_rows {
            for uj in 0..self.n_columns {
                let idx_linear = ui * self.n_columns + uj;
                let idx_thread = find_slot_index(
                    &self
                        .thread_pool
                        .iter()
                        .map(|x| x.scratch_offset)
                        .collect::<Vec<usize>>(),
                    &idx_linear,
                );
                let mut scratch_moments = self.scratch_moments[idx_thread].lock().unwrap();
                scratch_moments[idx_linear - self.thread_pool[idx_thread].scratch_offset] =
                    moments_read[(ui, uj)];
            }
        }
    }

    /// setter to write a new full matrix of moments to the lattice
    /// *and* update scratch regions
    pub fn set_moments(&mut self, new_moments: Matrix<f64>) -> () {
        {
            //scope for moments write lock; must end before copy_to_scratch
            let mut moments = self.moments.write().unwrap();
            for i in 0..self.n_rows {
                for j in 0..self.n_columns {
                    moments[(i, j)] = new_moments[(i, j)];
                }
            }
        }
        self.copy_to_scratch()
    }

    /// using a `Config` and a hamiltonian function, creates a new lattice.
    /// Creates and initializes arrays and the workers with their threads

    // XXX TODO: this also needs decomposition work.
    pub fn new(c: &Config, hamiltonian: F) -> Lattice<F>
    where
        F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send,
    {
        let n_rows = c.n_rows;
        let n_columns = c.n_columns;
        let mut moments_data: Vec<f64> = Vec::with_capacity(c.n_rows * c.n_columns);
        let mut temperature_data: Vec<f64> = Vec::with_capacity(c.n_rows * c.n_columns);
        let mut external_field_data: Vec<f64> = Vec::with_capacity(c.n_rows * c.n_columns);
        for _ in 0..c.n_rows * c.n_columns {
            let r: f64 = rand::random();
            if r > 0.5 {
                moments_data.push(-1.0 * c.magnetic_moment);
            } else {
                moments_data.push(c.magnetic_moment);
            }
            temperature_data.push(c.temperature);
            external_field_data.push(c.external_field);
        }
        let moments = Arc::new(RwLock::new(Matrix::new(
            c.n_rows,
            c.n_columns,
            moments_data,
        )));
        let temperature = Arc::new(RwLock::new(Matrix::new(
            c.n_rows,
            c.n_columns,
            temperature_data,
        )));
        let external_field = Arc::new(RwLock::new(Matrix::new(
            c.n_rows,
            c.n_columns,
            external_field_data,
        )));
        let n_threads = c.n_threads;

        let mut thread_chunk_sizes = vec![c.n_rows * c.n_columns];

        if n_threads > 1 {
            let is_round_multiple = (c.n_rows * c.n_columns) % (n_threads) == 0;
            if is_round_multiple {
                thread_chunk_sizes = vec![(c.n_rows * c.n_columns) / n_threads; n_threads];
            } else {
                thread_chunk_sizes =
                    vec![(c.n_rows * c.n_columns) / (n_threads - 1); n_threads - 1];
                thread_chunk_sizes.push((c.n_rows * c.n_columns) % (n_threads - 1));
            }
        }
        let scratch_barrier = Arc::new(Barrier::new(n_threads + 1)); //+1 for main thread
        let mut scratch_moments: Vec<Arc<Mutex<Vec<f64>>>> =
            Vec::with_capacity(thread_chunk_sizes.len());

        for idx_chunk in 0..thread_chunk_sizes.len() {
            let mut scratch_moment = Vec::with_capacity(thread_chunk_sizes[idx_chunk]);
            for _ in 0..thread_chunk_sizes[idx_chunk] {
                scratch_moment.push(0.0);
            }
            scratch_moments.push(Arc::new(Mutex::new(scratch_moment)));
        }

        let mut workers = Vec::with_capacity(n_threads);

        for id in 0..n_threads {
            let local_hamiltonian = hamiltonian.clone();
            let (sender, receiver) = mpsc::sync_channel::<UpdateMessage>(0);

            workers.push(UpdateWorker::new(
                id,
                sender,
                receiver,
                Arc::clone(&scratch_barrier),
                Arc::clone(&moments),
                Arc::clone(&temperature),
                Arc::clone(&external_field),
                Arc::clone(&scratch_moments[id]),
                id * thread_chunk_sizes[0], //correct even if last chunk is smaller
                local_hamiltonian,
            ));
        }
        let can_wait_scratch_barrier = false;
        let mut ret_val = Lattice {
            n_rows,
            n_columns,
            moments,
            temperature,
            external_field,
            scratch_moments,
            thread_pool: workers,
            hamiltonian,
            scratch_barrier,
            can_wait_scratch_barrier,
        };
        // now that the lattice exists, we can invoke methods on it to complete
        // init
        ret_val.copy_to_scratch();
        return ret_val;
    }

    /// uses the hamiltonian to calculate the energy of the lattice, in units of
    /// kBT or whatever your temperature is.  "Natural units."
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        let actual_moments = self.moments.read().unwrap();
        let actual_external_field = self.external_field.read().unwrap();
        for ui in 0..self.n_rows {
            for uj in 0..self.n_columns {
                energy += (self.hamiltonian)(
                    &actual_moments,
                    &actual_external_field,
                    &(ui as i64, uj as i64),
                );
            }
        }
        energy
    }

    /// similar to `energy`, but returns the net magnetization of the lattice
    /// which is just the sum of magnetic moments of all sites.
    pub fn net_magnetization(&self) -> f64 {
        let mut net_magnetization = 0.0;
        let actual_moments = self.moments.read().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                net_magnetization += actual_moments[(i, j)];
            }
        }
        return net_magnetization;
    }

    /// invokes plotly to displace lattice moments as an image.  Saves to disk
    /// and optionally displays to screen (via web browser).  Slow as hell!
    pub fn moments_as_heatmap(
        &self,
        filename: String,
        title_annotation: String,
        to_screen: bool,
    ) -> () {
        let trace = HeatMap::new_z(self.moments_as_vec_vec());

        let mut plot = Plot::new();
        let title = Title::new(&format!("Magnetic Moments {}", title_annotation));
        let layout = Layout::new().title(title).width(2000).height(2000);
        plot.add_trace(trace);
        plot.set_layout(layout);

        plot.write_image(filename, ImageFormat::PNG, 1600, 1600, 1.0);
        if to_screen {
            plot.show();
        }
    }
}

/// A hamiltonian function that is submitted to Lattice as a _member_,  not a
/// method because it can be selected at runtime and user defined.  
///
/// note that the function needs to return the PER SITE energy, so hamiltonian
/// is really the sum of this over all sites.
///
/// `mixin_map_hamiltonian` takes a rectangular "map" of interaction strengths
/// and *returns* a hamiltonian, rather than being a hamiltonian itself.
///
/// J_ij is a typical notation for the map of interaction strengths, as used at
/// the wikipedia link below.  This map must have odd dimensions, so the site
/// in question can be at the center of the map. For each site, the energy is calculated
/// by applying this map (which is likely much smaller than the lattice) to the surrounding
/// sites.  Thus the ising hamiltonian has a 3x3 map with ones at (0,1), (1,0), (1,2), (2,1)
/// and zeros elsewhere.  To be explicit, this makes the indices *relative* to the site,
/// not absolute positions on the lattice.  This makes the map small and relatively dense,
/// while J_ij with absolute i and j would be large and sparse.
/// # Example
/// This is a spin on the classic ising model, where there is ferromagnetic interaction
/// with nearest neighbors--but also a mild antiferromagnetic interaction with the next-nearest
/// neighbors.  It results in a structure that has a characteristic length scale of 2-3 sites,
/// which isn't much of a surprise.
/// ```
/// use ising_toy::*;
/// let c = load_config(None);
/// let hsize: (usize, usize) = (5, 5);
/// let mut interaction_strenghts = Matrix::new(hsize.0, hsize.1, vec![0.0; hsize.0 * hsize.1]);
/// interaction_strenghts[(2, 0)] = -0.5;
/// interaction_strenghts[(0, 2)] = -0.5;
/// interaction_strenghts[(2, 4)] = -0.5;
/// interaction_strenghts[(4, 2)] = -0.5;
/// interaction_strenghts[(1, 2)] = 1.0;
/// interaction_strenghts[(2, 1)] = 1.0;
/// interaction_strenghts[(3, 2)] = 1.0;
/// interaction_strenghts[(2, 3)] = 1.0;
/// let hamiltonian = mixin_map_hamiltonian(&interaction_strenghts);
/// let mut lattice = Lattice::new(&c, hamiltonian);
/// ```
/// # See also
/// * `Lattice::energy()`
/// * [Wikipedia](https://en.wikipedia.org/wiki/Ising_model)

pub fn mixin_map_hamiltonian(
    interaction_strengths: &Matrix<f64>,
) -> impl Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone {
    let my_map = interaction_strengths.clone();
    return move |moments: &Matrix<f64>, external_field: &Matrix<f64>, index: &(i64, i64)| -> f64 {
        let site_moment = &moments[moments.wrap(*index)];
        let mut energy = -(site_moment) * external_field[external_field.wrap(*index)];
        let offset = ((my_map.n_columns / 2) as i64, (my_map.n_rows / 2) as i64);
        for i in 0..my_map.n_rows {
            for j in 0..my_map.n_columns {
                energy += -(site_moment)
                    * my_map[(i, j)]
                    * moments[moments
                        .wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1))];
            }
        }
        return energy;
    };
}

/// A standard Ising model hamiltonian. It uses
/// `mixin_map_hamiltonian` with a map that has unit interaction strength for
/// sites sharing an edge and zero elsewhere.
pub fn standard_ising_hamiltonian(
) -> impl Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone {
    let hsize: (usize, usize) = (3, 3);
    let mut hmap = Matrix::new(hsize.0, hsize.1, vec![1.0; hsize.0 * hsize.1]);
    hmap[(0, 0)] = 0.0;
    hmap[(0, 2)] = 0.0;
    hmap[(2, 0)] = 0.0;
    hmap[(2, 2)] = 0.0;
    hmap[(1, 1)] = 0.0;
    return mixin_map_hamiltonian(&hmap);
}

/// A hamiltonian function for Conway's Game of Life.  This is a cellular automaton
/// as described by [wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).  
/// It's a silly "hamiltonian" but it does follow
/// the rules of the game of life, returning high energy when site should change
/// and low energy when it should stay the same.
/// # Rules
/// * Any live cell with fewer than two live neighbors dies, as if by underpopulation.
/// * Any live cell with two or three live neighbors lives on to the next generation.
/// * Any live cell with more than three live neighbors dies, as if by overpopulation.
/// * Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
/// note that this is to produce the ENERGY of the site, not the state of it.  we want the correct state to
/// have overwhelmingly less energy than the incorrect state, so `update` always selects it when it does its
/// probability check.
///
/// Interestingly, when at a finite temperature, this framework models the game
/// of life--with a chance for a cell to violate the rules.  It's stateless, though
/// so it's not really modeling mutations or anything: a cell that survives 4
/// neighbors one turn might die due to the same 4 neighbors the next.
///
/// Likewise, a background "magnetic field" here functions as a bias towards extra
/// cells growing or dying, in violation of the rules above.
/// # See also
/// [wikipedia](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
pub fn conway_life_hamiltonian(
    moments: &Matrix<f64>,
    background_field: &Matrix<f64>,
    index: &(i64, i64),
) -> f64 {
    let mut n_neighbors = 0;
    let offset = (1, 1);
    let mut site_state = 0.0; //always overwritten but compiler doesn't know that

    for i in 0..3 {
        for j in 0..3 {
            if i == 1 && j == 1 {
                site_state = moments[moments.wrap(*index)];
                continue; //don't add self to neighbors tally
            }
            //below converts -1/1 states to 0/1 neighbors. -1=dead
            let (ui, uj) =
                moments.wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1));
            let tmp = (moments[(ui, uj)] as i32 + 1) / 2;
            n_neighbors += tmp;
            //println!("site ({},{}) has {} cell(s)", ui, uj, tmp);
        }
    }
    let should_live = (n_neighbors == 3) || (n_neighbors == 2 && site_state > 0.0);

    if should_live {
        if site_state > 0.0 {
            return -1e0; //alive and should stay alive
        } else {
            return 1e0; //dead and should come alive
        }
    } else if site_state > 0.0 {
        return 1e0; //alive and should die
    } else {
        return -1e0; //dead and should stay dead
    }
}

/// Helpful for bulk edits of a matrix to use in some aspect of the simulation.
pub fn row_edit<T: Clone>(mat: &mut Matrix<T>, row: usize, values: Vec<T>) -> () {
    for i in 0..mat.n_columns {
        mat.data[row * mat.n_columns + i] = values[i].clone();
    }
}
/// Helpful for bulk edits of a matrix to use in some aspect of the simulation.
pub fn column_edit<T: Clone>(mat: &mut Matrix<T>, column: usize, values: Vec<T>) -> () {
    for i in 0..mat.n_rows {
        mat.data[i * mat.n_columns + column] = values[i].clone();
    }
}

/// helper function to find the index of a value in a sorted vector.  
/// Used for finding the right worker thread for a given site.
//XXX TODO: see if Vec has a built-in method for this
pub fn find_slot_index<T: PartialOrd>(vec: &Vec<T>, value: &T) -> usize {
    //note by implication anything beyond either end is put in first or last slot without limit
    let mut index = 0;
    for i in 1..vec.len() {
        //use of 1 above bins anything less than vec[1] into the first slot
        if vec[i] <= *value {
            index += 1;
        }
    }
    return index;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn broadcast_64() {
        let shape = (2, 2);
        let a: f64 = 1.0;
        let b: Matrix<f64> = a.broadcast(shape);
        let c: Matrix<f64> = Matrix::new(shape.0, shape.1, vec![1.0; 4]);
        let d: Matrix<f64> = c.broadcast((b.n_rows, b.n_columns));
        assert_eq!(b.data, vec![1.0; 4]);
        assert_eq!(c.data, vec![1.0; 4]);
        assert_eq!(d.data, vec![1.0; 4]);
        assert_eq!(b.n_rows, shape.0);
        assert_eq!(b.n_columns, shape.1);
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(matrix.n_rows, 2);
        assert_eq!(matrix.n_columns, 3);
        assert_eq!(matrix.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_matrix_indexing() {
        let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_index_mut() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]);
        matrix[(0, 0)] = 5;
        matrix[(1, 1)] = 6;
        assert_eq!(matrix[(0, 0)], 5);
        assert_eq!(matrix[(1, 1)], 6);
    }

    #[test]
    fn test_matrix_wrap() {
        let matrix = Matrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(matrix.wrap((-1, -1)), (2, 2));
        assert_eq!(matrix.wrap((3, 3)), (0, 0));
        assert_eq!(matrix.wrap((5, 5)), (2, 2));
    }
}
