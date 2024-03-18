use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};

use core::panic;
use std::ops::{Index, IndexMut};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
pub static DEFAULT_MAGNETIC_MOMENT: f64 = 1.0;
pub static DEFAULT_TEMPERATURE: f64 = 0.5;
pub static DEFAULT_EXTERNAL_FIELD: f64 = 0.0;
pub static N_ROWS: usize = 100;
pub static N_COLUMNS: usize = 100;

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

pub struct UpdateWorker {
    pub id: usize,
    pub thread: thread::JoinHandle<()>,
    pub moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    hamiltonian: fn(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>, (i64, i64)) -> f64,
    hamiltonian_map: Matrix<f64>,
}
impl UpdateWorker {
    pub fn new(
        id: usize,
        receiver: Arc<Mutex<mpsc::Receiver<UpdateLocation>>>,
        moments: Arc<RwLock<Matrix<f64>>>,
        temperature: Arc<RwLock<Matrix<f64>>>,
        external_field: Arc<RwLock<Matrix<f64>>>,
        hamiltonian: fn(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>, (i64, i64)) -> f64,
        hamiltonian_map: Matrix<f64>,
    ) -> UpdateWorker {
        let ref_copy_hamiltonian = hamiltonian.clone();
        let ref_copy_hamiltonian_map = hamiltonian_map.clone();
        let ref_copy_moments = Arc::clone(&moments);
        let ref_copy_temperature = Arc::clone(&temperature);
        let ref_copy_external_field = Arc::clone(&external_field);
        let thread = thread::spawn(move || loop {
            let update_loc = match receiver.lock().unwrap().recv() {
                Ok(update_loc) => update_loc,
                Err(_) => break,
            };

            //println!("Worker {id} got a job; executing.");
            let read_moments = moments.read().unwrap();
            let read_external_field = external_field.read().unwrap();
            let (ui, uj) = read_moments.wrap((update_loc.i, update_loc.j));
            let e_site = hamiltonian(
                &read_moments,
                &read_external_field,
                &hamiltonian_map,
                (update_loc.i, update_loc.j),
            );
            let t_site = temperature.read().unwrap()[(ui, uj)];
            let p_state =
                (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());

            let this_moment = read_moments[(ui, uj)];

            let r: f64 = rand::random();
            if r > p_state {
                *update_loc.scratch_moment.lock().unwrap() = this_moment * -1.0;
            } else {
                *update_loc.scratch_moment.lock().unwrap() = this_moment;
            }
        });
        UpdateWorker {
            id,
            thread,
            moments: ref_copy_moments,
            temperature: ref_copy_temperature,
            external_field: ref_copy_external_field,
            hamiltonian: ref_copy_hamiltonian,
            hamiltonian_map: ref_copy_hamiltonian_map,
        }
    }
}
//type Job = Box<dyn FnOnce() + Send + 'static>;
pub struct UpdateLocation {
    pub i: i64,
    pub j: i64,
    pub scratch_moment: Arc<Mutex<f64>>,
}

impl Clone for UpdateLocation {
    fn clone(&self) -> UpdateLocation {
        let i = self.i;
        let j = self.j;
        let scratch_moment = Arc::clone(&self.scratch_moment);
        UpdateLocation {
            i,
            j,
            scratch_moment,
        }
    }
}
pub struct AlternateLattice {
    pub n_rows: usize,
    pub n_columns: usize,
    pub moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    scratch_moments: Matrix<Arc<Mutex<f64>>>,
    pub sender: mpsc::SyncSender<UpdateLocation>,
    pub thread_pool: Vec<UpdateWorker>,
    pub local_hamiltonian: fn(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>, (i64, i64)) -> f64,
    pub hamiltonian_map: Matrix<f64>,
}

impl AlternateLattice {
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
    pub fn new_update(&mut self) -> () {
        //XXX TODO: needs to be synchronized before self.copy_from_scratch()
        //??? channels back to the main thread, saying when done?
        //shared bool array of "done"s, one for each thread?
        // `loop` closures will never be idle so can't check that, right?
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                let scratch_moment = Arc::clone(&self.scratch_moments[(i, j)]);
                let update_loc = UpdateLocation {
                    i: i as i64,
                    j: j as i64,
                    scratch_moment,
                };
                self.sender.send(update_loc).unwrap();
            }
        }
        self.copy_from_scratch()
    }
    pub fn sequential_update(&mut self) -> () {
        let mut handles = vec![];
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                let handle = self.update((i as i64, j as i64));
                handles.push(handle);
            }
        }
        // for handle in handles {
        //     handle.join().unwrap();
        // }

        self.copy_from_scratch();
    }
    fn copy_from_scratch(&mut self) -> () {
        let mut actual_moments = self.moments.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                actual_moments[(i, j)] = *self.scratch_moments[(i, j)].lock().unwrap();
            }
        }
    }

    pub fn update(&mut self, (i, j): (i64, i64)) -> () {
        let actual_moments = self.moments.read().unwrap();
        let (ui, uj) = actual_moments.wrap((i, j)).clone();
        let actual_external_field = self.external_field.read().unwrap();
        let mut scratch_moment = self.scratch_moments[(ui, uj)].lock().unwrap();

        let e_site = (self.local_hamiltonian)(
            &actual_moments,
            &actual_external_field,
            &self.hamiltonian_map,
            (i, j),
        );
        let t_site = self.temperature.read().unwrap()[(ui, uj)];
        let p_state =
            (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());
        let r: f64 = rand::random();

        if r > p_state {
            *scratch_moment = -1.0 * actual_moments[(ui, uj)];
        } else {
            *scratch_moment = actual_moments[(ui, uj)];
        }
    }
    pub fn new(
        n_rows: usize,
        n_columns: usize,
        local_hamiltonian: fn(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>, (i64, i64)) -> f64,
        hamiltonian_map: Matrix<f64>,
    ) -> AlternateLattice {
        let mut moments_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut scratch_moments: Vec<Arc<Mutex<f64>>> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut temperature_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut external_field_data: Vec<f64> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        for _ in 0..N_COLUMNS * N_ROWS {
            let r: f64 = rand::random();
            if r > 0.5 {
                moments_data.push(-1.0 * DEFAULT_MAGNETIC_MOMENT);
            } else {
                moments_data.push(DEFAULT_MAGNETIC_MOMENT);
            }
            scratch_moments.push(Arc::new(Mutex::new(0.0)));
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
        let mut workers = Vec::with_capacity(n_threads);
        let (sender, receiver) = mpsc::sync_channel(12);
        let receiver = Arc::new(Mutex::new(receiver));
        for id in 0..n_threads {
            workers.push(UpdateWorker::new(
                id,
                Arc::clone(&receiver),
                Arc::clone(&moments),
                Arc::clone(&temperature),
                Arc::clone(&external_field),
                local_hamiltonian.clone(),
                hamiltonian_map.clone(),
            ));
        }

        return AlternateLattice {
            n_rows,
            n_columns,
            moments,
            temperature,
            external_field,
            local_hamiltonian,
            hamiltonian_map,
            scratch_moments: Matrix::new(N_ROWS, N_COLUMNS, scratch_moments),
            sender,
            thread_pool: workers,
        };
    }
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        let actual_moments = self.moments.read().unwrap();
        let actual_external_field = self.external_field.read().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                energy += (self.local_hamiltonian)(
                    &actual_moments,
                    &actual_external_field,
                    &self.hamiltonian_map,
                    (i as i64, j as i64),
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
pub fn a_hamiltonian(
    moments: &Matrix<f64>,
    external_field: &Matrix<f64>,
    hamiltonian_map: &Matrix<f64>,
    index: (i64, i64),
) -> f64 {
    let site_moment = &moments[moments.wrap(index)];
    let mut energy = -(site_moment) * external_field[external_field.wrap(index)];
    let offset = (
        (hamiltonian_map.n_cols / 2) as i64,
        (hamiltonian_map.n_rows / 2) as i64,
    );

    for i in 0..hamiltonian_map.n_rows {
        for j in 0..hamiltonian_map.n_cols {
            energy += -(site_moment)
                * hamiltonian_map[(i, j)]
                * moments
                    [moments.wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1))];
        }
    }
    return energy;
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
