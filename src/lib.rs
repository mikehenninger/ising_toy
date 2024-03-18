use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};
use std::fs::copy;
use std::ops::{Index, IndexMut};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
pub static DEFAULT_MAGNETIC_MOMENT: f64 = 1.0;
pub static DEFAULT_TEMPERATURE: f64 = 0.5;
pub static DEFAULT_EXTERNAL_FIELD: f64 = 0.0;
pub static N_ROWS: usize = 20;
pub static N_COLUMNS: usize = 4;

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

pub struct LatticeSite {
    pub magnetic_moment: f64,
    pub temperature: f64,
    pub external_field: f64,
}

impl Clone for LatticeSite {
    fn clone(&self) -> LatticeSite {
        LatticeSite {
            magnetic_moment: self.magnetic_moment,
            temperature: self.temperature,
            external_field: self.external_field,
        }
    }
}

impl LatticeSite {
    pub fn new(magnetic_moment: f64, temperature: f64, external_field: f64) -> LatticeSite {
        LatticeSite {
            magnetic_moment,
            temperature,
            external_field,
        }
    }
}

pub struct Lattice {
    pub n_rows: usize,
    pub n_columns: usize,
    pub sites: Arc<RwLock<Matrix<LatticeSite>>>,
    pub local_hamiltonian: fn(&Matrix<LatticeSite>, &Matrix<f64>, (i64, i64)) -> f64,
    pub hamiltonian_map: Matrix<f64>,
    pub scratch_sites: Matrix<Arc<Mutex<LatticeSite>>>,
}
impl Lattice {
    pub fn set_temperature<T>(&mut self, temperature: T) -> ()
    where
        T: Broadcast,
    {
        let temp = temperature.broadcast((self.n_rows, self.n_columns));
        let mut actual_sites = self.sites.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                actual_sites[(i, j)].temperature = temp[(i, j)];
            }
        }
    }
    pub fn set_external_field<T>(&mut self, external_field: T) -> ()
    where
        T: Broadcast,
    {
        let field = external_field.broadcast((self.n_rows, self.n_columns));
        let mut actual_sites = self.sites.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                actual_sites[(i, j)].external_field = field[(i, j)];
            }
        }
    }

    fn moments_as_vec_vec(&self) -> Vec<Vec<f64>> {
        let mut vec_vec = Vec::new();
        let actual_sites = self.sites.read().unwrap();
        for i in 0..self.n_rows {
            let mut row = Vec::new();
            for j in 0..self.n_columns {
                row.push(actual_sites[(i, j)].magnetic_moment);
            }
            vec_vec.push(row);
        }
        vec_vec
    }
    pub fn sequential_update(&mut self) -> () {
        let mut handles = vec![];
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                let handle = self.update_scratch((i as i64, j as i64));
                handles.push(handle);
            }
        }
        for handle in handles {
            handle.join().unwrap();
        }

        self.copy_from_scratch();
    }

    pub fn update(&mut self, (i, j): (i64, i64)) -> () {
        let mut actual_sites = self.sites.write().unwrap();
        let (ui, uj) = actual_sites.wrap((i, j)).clone();
        let e_site = (self.local_hamiltonian)(&actual_sites, &self.hamiltonian_map, (i, j));
        let t_site = self.sites.read().unwrap()[(ui, uj)].temperature;
        let p_state =
            (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());
        let r: f64 = rand::random();

        if r > p_state {
            actual_sites[(ui, uj)].magnetic_moment *= -1.0;
        }
    }

    pub fn update_scratch(&mut self, (i, j): (i64, i64)) -> JoinHandle<()> {
        let guarded_sites = Arc::clone(&self.sites);
        let the_ham = self.local_hamiltonian.clone();
        let the_map = self.hamiltonian_map.clone();
        let scratch_site_to_thread = Arc::clone(&self.scratch_sites[(i as usize, j as usize)]);
        let handle = thread::spawn(move || {
            let actual_sites = guarded_sites.read().unwrap();
            let (ui, uj) = actual_sites.wrap((i, j));
            let e_site = the_ham(&actual_sites, &the_map, (i, j));
            let t_site = actual_sites[(ui, uj)].temperature;
            let p_state =
                (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());

            let moment_to_thread = actual_sites[(ui, uj)].magnetic_moment;

            let r: f64 = rand::random();
            if r > p_state {
                scratch_site_to_thread.lock().unwrap().magnetic_moment = moment_to_thread * -1.0;
            } else {
                scratch_site_to_thread.lock().unwrap().magnetic_moment = moment_to_thread;
            }
        });
        return handle;
    }
    fn copy_from_scratch(&mut self) -> () {
        let mut actual_sites = self.sites.write().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                // println!(
                //     "before:{},{}",
                //     self.sites[(i, j)].magnetic_moment,
                //     self.scratch_sites[(i, j)].lock().unwrap().magnetic_moment
                // );
                actual_sites[(i, j)].magnetic_moment =
                    self.scratch_sites[(i, j)].lock().unwrap().magnetic_moment;
                // println!(
                //     "after:{},{}",
                //     self.sites[(i, j)].magnetic_moment,
                //     self.scratch_sites[(i, j)].lock().unwrap().magnetic_moment
                // );
            }
        }
    }
    pub fn moments_as_heatmap(&self) -> () {
        let trace = HeatMap::new_z(self.moments_as_vec_vec());

        let mut plot = Plot::new();
        let title = Title::new("Magnetic Moments");
        let layout = Layout::new().title(title).width(800).height(800);
        plot.add_trace(trace);
        plot.set_layout(layout);
        plot.show();
        plot.write_image("out.png", ImageFormat::PNG, 800, 600, 1.0);
    }
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;
        let actual_sites = self.sites.read().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                energy += (self.local_hamiltonian)(
                    &actual_sites,
                    &self.hamiltonian_map,
                    (i as i64, j as i64),
                );
            }
        }
        energy
    }

    pub fn new(
        n_rows: usize,
        n_columns: usize,
        local_hamiltonian: fn(&Matrix<LatticeSite>, &Matrix<f64>, (i64, i64)) -> f64,
        hamiltonian_map: Matrix<f64>,
    ) -> Lattice {
        let mut all_sites: Vec<LatticeSite> = Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut scratch_sites: Vec<Arc<Mutex<LatticeSite>>> =
            Vec::with_capacity(N_ROWS * N_COLUMNS);
        let mut r: f64;
        for _ in 0..N_COLUMNS * N_ROWS {
            r = rand::random();
            let mut sign = 1.0;
            if r > 0.5 {
                sign = -1.0;
            }
            all_sites.push(LatticeSite::new(
                sign * DEFAULT_MAGNETIC_MOMENT,
                DEFAULT_TEMPERATURE,
                DEFAULT_EXTERNAL_FIELD,
            ));

            scratch_sites.push(Arc::new(Mutex::new(LatticeSite::new(0.0, 0.0, 0.0))));
        }

        return Lattice {
            n_rows,
            n_columns,
            sites: Arc::new(RwLock::new(Matrix::new(N_ROWS, N_COLUMNS, all_sites))),
            local_hamiltonian,
            hamiltonian_map,
            scratch_sites: Matrix::new(N_ROWS, N_COLUMNS, scratch_sites),
        };
    }
    pub fn net_magnetization(&self) -> f64 {
        let mut net_magnetization = 0.0;
        let actual_sites = self.sites.read().unwrap();
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                net_magnetization += actual_sites[(i, j)].magnetic_moment;
            }
        }
        return net_magnetization;
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
    sites: &Matrix<LatticeSite>,
    hamiltonian_map: &Matrix<f64>,
    index: (i64, i64),
) -> f64 {
    let site = &sites[sites.wrap(index)];
    let mut energy = -(site.magnetic_moment) * site.external_field;
    let offset = (
        (hamiltonian_map.n_cols / 2) as i64,
        (hamiltonian_map.n_rows / 2) as i64,
    );

    for i in 0..hamiltonian_map.n_rows {
        for j in 0..hamiltonian_map.n_cols {
            energy += -(site.magnetic_moment)
                * hamiltonian_map[(i, j)]
                * sites[sites.wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1))]
                    .magnetic_moment;
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
