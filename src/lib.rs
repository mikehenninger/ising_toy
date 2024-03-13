use std::ops::{Index, IndexMut};

use plotly::common::Title;
use plotly::{HeatMap, ImageFormat, Layout, Plot};
pub static DEFAULT_MAGNETIC_MOMENT: f64 = 1.0;
pub static DEFAULT_TEMPERATURE: f64 = 0.5;
pub static DEFAULT_EXTERNAL_FIELD: f64 = -0.0;
pub static N_ROWS: usize = 150;
pub static N_COLUMNS: usize = 150;

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
    pub sites: Matrix<LatticeSite>,
    pub local_hamiltonian: fn(&Lattice, &Matrix<f64>, (i64, i64)) -> f64,
    pub hamiltonian_map: Matrix<f64>,
}
impl Lattice {
    pub fn set_temperature<T>(&mut self, temperature: T) -> ()
    where
        T: Broadcast,
    {
        let temp = temperature.broadcast((self.n_rows, self.n_columns));
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                self.sites[(i, j)].temperature = temp[(i, j)];
            }
        }
    }
    pub fn set_external_field<T>(&mut self, external_field: T) -> ()
    where
        T: Broadcast,
    {
        let field = external_field.broadcast((self.n_rows, self.n_columns));
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                self.sites[(i, j)].external_field = field[(i, j)];
            }
        }
    }
    fn moments_as_vec_vec(&self) -> Vec<Vec<f64>> {
        let mut vec_vec = Vec::new();
        for i in 0..self.n_rows {
            let mut row = Vec::new();
            for j in 0..self.n_columns {
                row.push(self.sites[(i, j)].magnetic_moment);
            }
            vec_vec.push(row);
        }
        vec_vec
    }
    pub fn sequential_update(&mut self) -> () {
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                self.update((i as i64, j as i64));
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
        for i in 0..self.n_rows {
            for j in 0..self.n_columns {
                energy +=
                    (self.local_hamiltonian)(&self, &self.hamiltonian_map, (i as i64, j as i64));
            }
        }
        energy
    }
    pub fn update(&mut self, (i, j): (i64, i64)) -> () {
        let (ui, uj) = self.sites.wrap((i, j));
        let e_site = (self.local_hamiltonian)(&self, &self.hamiltonian_map, (i, j));
        let t_site = self.sites[(ui, uj)].temperature;
        let p_state =
            (-e_site / t_site).exp() / ((-e_site / t_site).exp() + (e_site / t_site).exp());
        let r: f64 = rand::random();
        //println!("p: {}, e: {}", p_state, e_site);
        if r > p_state {
            self.sites[(ui, uj)].magnetic_moment = -self.sites[(ui, uj)].magnetic_moment;
        }
    }
    pub fn new(
        n_rows: usize,
        n_columns: usize,
        local_hamiltonian: fn(&Self, &Matrix<f64>, (i64, i64)) -> f64,
        hamiltonian_map: Matrix<f64>,
    ) -> Lattice {
        let mut all_sites: Vec<LatticeSite> = Vec::new();
        let mut r: f64;
        for _ in 0..N_COLUMNS * N_ROWS {
            r = rand::random();
            if r > 0.5 {
                all_sites.push(LatticeSite::new(
                    DEFAULT_MAGNETIC_MOMENT,
                    DEFAULT_TEMPERATURE,
                    DEFAULT_EXTERNAL_FIELD,
                ));
            } else {
                all_sites.push(LatticeSite::new(
                    -DEFAULT_MAGNETIC_MOMENT,
                    DEFAULT_TEMPERATURE,
                    DEFAULT_EXTERNAL_FIELD,
                ));
            }
        }

        return Lattice {
            n_rows,
            n_columns,
            sites: Matrix::new(N_ROWS, N_COLUMNS, all_sites),
            local_hamiltonian,
            hamiltonian_map,
        };
    }
}

/// A hamiltonian function that is not a method because it can be swapped out
/// note that this returns the PER SITE energy, so hamiltonian is really the
/// sum of this.
/// # See also
/// `Lattice::energy()`
/// * [Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
// TODO: probably make this one of an enum of hamiltonians
pub fn a_hamiltonian(lattice: &Lattice, hamiltonian_map: &Matrix<f64>, index: (i64, i64)) -> f64 {
    let site = &lattice.sites[lattice.sites.wrap(index)];
    let mut energy = -(site.magnetic_moment) * site.external_field;
    let offset = (
        (hamiltonian_map.n_cols / 2) as i64,
        (hamiltonian_map.n_rows / 2) as i64,
    );

    for i in 0..hamiltonian_map.n_rows {
        for j in 0..hamiltonian_map.n_cols {
            energy += -(site.magnetic_moment)
                * hamiltonian_map[(i, j)]
                * lattice.sites[lattice
                    .sites
                    .wrap((i as i64 + index.0 - offset.0, j as i64 + index.1 - offset.1))]
                .magnetic_moment;
        }
    }
    return energy;
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
}
