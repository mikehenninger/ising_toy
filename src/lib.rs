// turn below off for final checking
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_mut)]

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

use core::panic;
use std::ops::{Index, IndexMut};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;
pub static DEFAULT_MAGNETIC_MOMENT: f64 = 1.0;
pub static DEFAULT_TEMPERATURE: f64 = 0.5;
pub static DEFAULT_EXTERNAL_FIELD: f64 = 0.0;
pub static N_ROWS: usize = 1000;
pub static N_COLUMNS: usize = 500;

pub trait Broadcast {
    fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64>;
}

impl Broadcast for f64 {
    fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64> {
        return Matrix::new(shape.0, shape.1, vec![*self; shape.0 * shape.1]);
    }
}

impl Broadcast for Matrix<f64> {
    fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64> {
        assert_eq!(self.n_rows, shape.0);
        assert_eq!(self.n_cols, shape.1);
        return self.clone();
    }
}
pub struct Matrix<T: Clone> {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<T>,
}

impl<T: Clone> Matrix<T> {
    pub fn as_vec_vec(&self) -> Vec<Vec<T>> {
        let mut vec_vec = Vec::new();
        for i in 0..self.n_rows {
            //TODO: VECTORIZE THIS
            let mut row = Vec::new();
            for j in 0..self.n_cols {
                row.push((*self.index((i, j))).clone());
            }
            vec_vec.push(row);
        }
        vec_vec
    }
    pub fn new(n_rows: usize, n_cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            n_rows * n_cols,
            data.len(),
            "Data does not match dimensions"
        );
        Self {
            n_rows,
            n_cols,
            data,
        }
    }
    pub fn wrap(&self, index: (i64, i64)) -> (usize, usize) {
        let mut i = index.0;
        let mut j = index.1;
        // % doesn't modulo negative numbers, so we wrap by hand
        while i < 0 {
            i += self.n_rows as i64;
        }
        while j < 0 {
            j += self.n_cols as i64;
        }
        i = i % self.n_rows as i64;
        j = j % self.n_cols as i64;
        (i as usize, j as usize)
    }
}

impl<T: Clone> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < self.n_rows && index.1 < self.n_cols,
            "Index out of bounds"
        );
        &self.data[index.0 * self.n_cols + index.1]
    }
}

impl<T: Clone> IndexMut<(usize, usize)> for Matrix<T> {
    //type Output = T;

    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(
            index.0 < self.n_rows && index.1 < self.n_cols,
            "Index out of bounds"
        );
        &mut self.data[index.0 * self.n_cols + index.1]
    }
}

impl<T> Clone for Matrix<T>
where
    T: Clone,
{
    fn clone(&self) -> Matrix<T> {
        Matrix {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data: self.data.clone(),
        }
    }
}

pub struct UpdateWorker<F: Clone + Send> {
    pub id: usize,
    pub thread: thread::JoinHandle<()>,
    pub moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    pub scratch_region: Arc<Mutex<Vec<f64>>>,
    pub scratch_offset: usize,
    sender: mpsc::SyncSender<UpdateMessage>,
    hamiltonian: F,
}
impl<F: Clone + Send> UpdateWorker<F> {
    pub fn new(
        id: usize,
        sender: mpsc::SyncSender<UpdateMessage>, //NOT moved into the thread
        receiver: mpsc::Receiver<UpdateMessage>,
        moments: Arc<RwLock<Matrix<f64>>>,
        temperature: Arc<RwLock<Matrix<f64>>>,
        external_field: Arc<RwLock<Matrix<f64>>>,
        scratch_region: Arc<Mutex<Vec<f64>>>,
        scratch_offset: usize,
        hamiltonian: F,
    ) -> UpdateWorker<F>
    where
        F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send + 'static,
    {
        let ref_copy_hamiltonian = hamiltonian.clone();
        let ref_copy_moments = Arc::clone(&moments);
        let ref_copy_temperature = Arc::clone(&temperature);
        let ref_copy_external_field = Arc::clone(&external_field);
        let ref_copy_scratch_region = Arc::clone(&scratch_region);
        let thread = thread::Builder::new()
            .spawn(move || loop {
                let update_message = match receiver.recv() {
                    Ok(update_message) => update_message,
                    Err(_) => break,
                };
                let read_moments = moments.read().expect("Could not read moments");
                //println!("Worker {id} has read lock on moments");
                let read_external_field = external_field
                    .read()
                    .expect("Could not read external field");

                let mut update_loc = match update_message {
                    UpdateMessage::Location(i, j) => vec![(i as i64, j as i64)],
                    UpdateMessage::All => {
                        let mut all_locs = Vec::new();
                        for idx_absolute in
                            scratch_offset..(scratch_offset + scratch_region.lock().unwrap().len())
                        {
                            all_locs.push((
                                (idx_absolute / read_moments.n_cols) as i64,
                                (idx_absolute % read_moments.n_cols) as i64,
                            ));
                        }
                        all_locs
                    }
                    UpdateMessage::Stop => break, //kill the spawn loop and thus the thread
                };
                //println!("Worker {id} got a job; executing.");
                {
                    // mutex locked scope
                    //println!("Worker {id} about to acquire mutex on its scratch region");
                    let mut scratch_region_locked =
                        scratch_region.lock().expect("couldn't lock scratch region");
                    //println!("Worker {id} has locked mutex on its scratch region");
                    for (i, j) in update_loc {
                        //println!("Worker {id} has read lock on external field");
                        let (ui, uj) = read_moments.wrap((i, j));
                        //println!("Worker {id} is updating ({i}, {j})/({ui}, {uj})");

                        //The error is here with scratch_offset being larger and trying to make usize negative

                        let offset_linear_loc = ui * read_moments.n_cols + uj - scratch_offset;

                        let e_site = hamiltonian(&read_moments, &read_external_field, &(i, j));
                        let t_site = temperature.read().unwrap()[(ui, uj)];
                        //println!("Worker {id} got and released lock on temperature");
                        let p_state = (-e_site / t_site).exp()
                            / ((-e_site / t_site).exp() + (e_site / t_site).exp());

                        let this_moment = read_moments[(ui, uj)];

                        let r: f64 = rand::random();
                        if r > p_state {
                            scratch_region_locked[offset_linear_loc] = -this_moment;
                        } else {
                            scratch_region_locked[offset_linear_loc] = this_moment;
                        }
                    }
                }
            })
            .expect("Could not create thread");
        UpdateWorker {
            id,
            sender,
            thread,
            moments: ref_copy_moments,
            temperature: ref_copy_temperature,
            external_field: ref_copy_external_field,
            hamiltonian: ref_copy_hamiltonian,
            scratch_region: ref_copy_scratch_region,
            scratch_offset, //survives all the way to here via copy
        }
    }
}
//type Job = Box<dyn FnOnce() + Send + 'static>;
pub enum UpdateMessage {
    Location(usize, usize),
    All,
    Stop,
}

// impl Clone for UpdateLocation {
//     fn clone(&self) -> UpdateLocation {
//         let i = self.i;
//         let j = self.j;
//         let scratch_moment = Arc::clone(&self.scratch_moment);
//         UpdateLocation {
//             i,
//             j,
//             scratch_moment,
//         }
//     }
// }
pub struct AlternateLattice<F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send> {
    pub n_rows: usize,
    pub n_columns: usize,
    pub moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    scratch_moments: Vec<Arc<Mutex<Vec<f64>>>>,
    //pub sender: mpsc::SyncSender<UpdateMessage>,
    pub thread_pool: Vec<UpdateWorker<F>>,
    // pub local_hamiltonian: fn(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>, (i64, i64)) -> f64,
    // pub hamiltonian_map: Matrix<f64>,
    pub hamiltonian: F,
}

impl<F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send + 'static>
    AlternateLattice<F>
{
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

    fn moments_as_vec_vec(&self) -> Vec<Vec<f64>> {
        let mut vec_vec = Vec::new();
        let actual_moments = self.moments.read().unwrap();
        for i in 0..self.n_rows {
            let mut row = Vec::new();
            for j in 0..self.n_columns {
                row.push(actual_moments[(i, j)]);
            }
            vec_vec.push(row);
        }
        vec_vec
    }

    pub fn full_update(&mut self) -> () {
        for worker in self.thread_pool.iter() {
            worker.sender.send(UpdateMessage::All).unwrap();
        }
        sleep(Duration::from_millis(5));
        self.copy_from_scratch();
    }

    pub fn incremental_update(&mut self) -> () {
        //XXX TODO: needs to be synchronized before self.copy_from_scratch()
        //??? channels back to the main thread, saying when done?
        //shared bool array of "done"s, one for each thread?
        // `loop` closures will never be idle so can't check that, right?
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
        // XXX THIS NEEDS A REAL SYNC
        //without this, it deadlocks between threads and copy_from_scratch
        //which I don't understand... a race, sure, but a deadlock???
        sleep(Duration::from_millis(5));
        //println!("All threads have been sent their updates; copying from scratch");
        self.copy_from_scratch();
        //println!("Done copying from scratch");
    }

    // pub fn sequential_update(&mut self) -> () {
    //     let mut handles = vec![];
    //     for i in 0..self.n_rows {
    //         for j in 0..self.n_columns {
    //             let handle = self.update((i as i64, j as i64));
    //             handles.push(handle);
    //         }
    //     }
    //     // for handle in handles {
    //     //     handle.join().unwrap();
    //     // }

    //     self.copy_from_scratch();
    // }
    fn copy_from_scratch(&mut self) -> () {
        //println!("Copying from scratch acquiring moments moments");
        let mut actual_moments = self.moments.write().unwrap();

        for idx_thread in 0..self.thread_pool.len() {
            let scratch_moments = self.scratch_moments[idx_thread].lock().unwrap();
            let scratch_offset = self.thread_pool[idx_thread].scratch_offset;
            for scratch_loc in 0..scratch_moments.len() {
                let absolute_loc = scratch_loc + scratch_offset;
                let (ui, uj) = (absolute_loc / self.n_columns, absolute_loc % self.n_columns);
                actual_moments[(ui, uj)] = scratch_moments[scratch_loc];
            }
        }
        //println!("Copying from scratch releasing moments moments");
    }

    // pub fn update(&mut self, (i, j): (i64, i64)) -> () {
    //     let actual_moments = self.moments.read().unwrap();
    //     let (ui, uj) = actual_moments.wrap((i, j)).clone();
    //     let actual_external_field = self.external_field.read().unwrap();
    //     let mut scratch_moment = self.scratch_moments[(ui, uj)].lock().unwrap();

    //     let e_site = (self.hamiltonian)(&actual_moments, &actual_external_field, &(i, j));
    //     let t_site = self.temperature.read().unwrap()[(ui, uj)];
    //     let p_state =
    //         (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());
    //     let r: f64 = rand::random();

    //     if r > p_state {
    //         *scratch_moment = -1.0 * actual_moments[(ui, uj)];
    //     } else {
    //         *scratch_moment = actual_moments[(ui, uj)];
    //     }
    // }
    pub fn new(n_rows: usize, n_columns: usize, hamiltonian: F) -> AlternateLattice<F>
    where
        F: Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone + Send,
    {
        let mut moments_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut temperature_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut external_field_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        for _ in 0..N_COLUMNS * N_ROWS {
            let r: f64 = rand::random();
            if r > 0.5 {
                moments_data.push(-1.0 * DEFAULT_MAGNETIC_MOMENT);
            } else {
                moments_data.push(DEFAULT_MAGNETIC_MOMENT);
            }
            temperature_data.push(DEFAULT_TEMPERATURE);
            external_field_data.push(DEFAULT_EXTERNAL_FIELD);
        }
        let moments = Arc::new(RwLock::new(Matrix::new(N_ROWS, N_COLUMNS, moments_data)));
        let temperature = Arc::new(RwLock::new(Matrix::new(
            N_ROWS,
            N_COLUMNS,
            temperature_data,
        )));
        let external_field = Arc::new(RwLock::new(Matrix::new(
            N_ROWS,
            N_COLUMNS,
            external_field_data,
        )));
        let n_threads = 24;

        let mut thread_chunk_sizes = vec![N_ROWS * N_COLUMNS];

        if n_threads > 1 {
            let is_multiple = (N_ROWS * N_COLUMNS) % (n_threads) == 0;
            if is_multiple {
                thread_chunk_sizes = vec![(N_ROWS * N_COLUMNS) / n_threads; n_threads];
            } else {
                thread_chunk_sizes = vec![(N_ROWS * N_COLUMNS) / (n_threads - 1); n_threads - 1];
                thread_chunk_sizes.push((N_ROWS * N_COLUMNS) % (n_threads - 1));
            }
        }

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
                Arc::clone(&moments),
                Arc::clone(&temperature),
                Arc::clone(&external_field),
                Arc::clone(&scratch_moments[id]),
                id * thread_chunk_sizes[0], //correct even if last chunk is smaller
                local_hamiltonian,
            ));
        }

        return AlternateLattice {
            n_rows,
            n_columns,
            moments,
            temperature,
            external_field,
            scratch_moments,
            thread_pool: workers,
            hamiltonian,
        };
    }
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        let actual_moments = self.moments.read().unwrap();
        let actual_external_field = self.external_field.read().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                energy += (self.hamiltonian)(
                    &actual_moments,
                    &actual_external_field,
                    &(i as i64, j as i64),
                );
            }
        }
        energy
    }
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
    pub fn moments_as_heatmap(&self, filename: String, to_screen: bool) -> () {
        let trace = HeatMap::new_z(self.moments_as_vec_vec());

        let mut plot = Plot::new();
        let title = Title::new("Magnetic Moments");
        let layout = Layout::new().title(title).width(800).height(800);
        plot.add_trace(trace);
        plot.set_layout(layout);

        plot.write_image(filename, ImageFormat::PNG, 800, 600, 1.0);
        if to_screen {
            plot.show();
        }
    }
}
/// A hamiltonian function that is not a method because it can be swapped out
/// note that this returns the PER SITE energy, so hamiltonian is really the
/// sum of this.
/// # See also
/// `Lattice::energy()`
/// * [Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
// TODO: probably make this one of an enum of hamiltonians

pub fn mapped_hamiltonian(
    hamiltonian_map: &Matrix<f64>,
) -> impl Fn(&Matrix<f64>, &Matrix<f64>, &(i64, i64)) -> f64 + Clone {
    let my_map = hamiltonian_map.clone();
    return move |moments: &Matrix<f64>, external_field: &Matrix<f64>, index: &(i64, i64)| -> f64 {
        let site_moment = &moments[moments.wrap(*index)];
        let mut energy = -(site_moment) * external_field[external_field.wrap(*index)];
        let offset = ((my_map.n_cols / 2) as i64, (my_map.n_rows / 2) as i64);
        for i in 0..my_map.n_rows {
            for j in 0..my_map.n_cols {
                energy += -(site_moment)
                    * my_map[(i, j)]
                    * moments[moments
                        .wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1))];
            }
        }
        return energy;
    };
}

pub fn row_edit<T: Clone>(mat: &mut Matrix<T>, row: usize, values: Vec<T>) -> () {
    for i in 0..mat.n_cols {
        mat.data[row * mat.n_cols + i] = values[i].clone();
    }
}

pub fn column_edit<T: Clone>(mat: &mut Matrix<T>, column: usize, values: Vec<T>) -> () {
    for i in 0..mat.n_rows {
        mat.data[i * mat.n_cols + column] = values[i].clone();
    }
}

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
        let d: Matrix<f64> = c.broadcast((b.n_rows, b.n_cols));
        assert_eq!(b.data, vec![1.0; 4]);
        assert_eq!(c.data, vec![1.0; 4]);
        assert_eq!(d.data, vec![1.0; 4]);
        assert_eq!(b.n_rows, shape.0);
        assert_eq!(b.n_cols, shape.1);
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(matrix.n_rows, 2);
        assert_eq!(matrix.n_cols, 3);
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
