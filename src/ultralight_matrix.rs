pub mod matrix {
    use std::ops::{Index, IndexMut};

    /// A trait for broadcasting a value or a matrix to a given shape.

    pub trait Broadcast {
        /// Broadcasts a value to the specified shape,
        /// returning a matrix with the specified shape,
        fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64>;
    }

    /// Implementation of the `Broadcast` trait for the `f64` type.
    /// fills each element with the f64 value that is self.
    /// Only needed this, so I haven't implemented any other types even though
    /// it's straightforward for this bare bones functionality

    impl Broadcast for f64 {
        fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64> {
            return Matrix::new(shape.0, shape.1, vec![*self; shape.0 * shape.1]);
        }
    }

    /// Implementation of the `Broadcast` trait for the `Matrix<f64>` type.
    /// Bare bones: it just checks that the shape is right then sends on a copy.
    impl Broadcast for Matrix<f64> {
        fn broadcast(&self, shape: (usize, usize)) -> Matrix<f64> {
            assert_eq!(self.n_rows, shape.0);
            assert_eq!(self.n_columns, shape.1);
            return self.clone();
        }
    }

    /// Represents a matrix of elements of type `T`.
    /// The matrix is stored as a flat vector in row-major order.
    /// Super bare bones, but since I wanted to add a minimal set of crates while learning
    /// I did it myself, implementing just what I needed
    /// I also assume there are real crates to do this, which I'd look for if I wasn't
    /// doing a purely learning exercise
    pub struct Matrix<T: Clone> {
        pub n_rows: usize,
        pub n_columns: usize,
        pub data: Vec<T>,
    }

    impl<T: Clone> Matrix<T> {
        /// Converts a copy of the matrix to a vector of vectors in the shape of the matrix
        /// Each inner vector represents a row of the matrix.
        /// used by plotly for 2d maps
        pub fn copy_as_vec_vec(&self) -> Vec<Vec<T>> {
            let mut vec_vec = Vec::new();
            for i in 0..self.n_rows {
                let mut row = Vec::new();
                for j in 0..self.n_columns {
                    row.push((*self.index((i, j))).clone());
                }
                vec_vec.push(row);
            }
            vec_vec
        }

        /// Creates a new matrix with the specified number of rows, columns, and data.
        /// The length of the data vector must match the product of the number of rows and columns.

        pub fn new(n_rows: usize, n_cols: usize, data: Vec<T>) -> Self {
            assert_eq!(
                n_rows * n_cols,
                data.len(),
                "Data does not match dimensions"
            );
            Self {
                n_rows,
                n_columns: n_cols,
                data,
            }
        }

        /// Wraps the given index to handle negative indices and indices larger than the matrix dimensions.
        /// Returns the wrapped index as a tuple of row and column indices.
        /// This is really handy because updating requires the hamiltonian probing
        /// nearby sites, and this way it wraps around the edges cleanly, not barfing
        /// on negative indices or indices larger than the matrix dimensions.
        pub fn wrap(&self, index: (i64, i64)) -> (usize, usize) {
            let mut i = index.0;
            let mut j = index.1;
            while i < 0 {
                i += self.n_rows as i64;
            }
            while j < 0 {
                j += self.n_columns as i64;
            }
            i = i % self.n_rows as i64;
            j = j % self.n_columns as i64;
            (i as usize, j as usize)
        }
    }

    /// Indexing implementation for the `Matrix` type.
    /// Allows accessing elements of the matrix using the `(row, column)` index.
    impl<T: Clone> Index<(usize, usize)> for Matrix<T> {
        type Output = T;

        fn index(&self, index: (usize, usize)) -> &Self::Output {
            assert!(
                index.0 < self.n_rows && index.1 < self.n_columns,
                "Index out of bounds"
            );
            &self.data[index.0 * self.n_columns + index.1]
        }
    }

    /// Mutable indexing implementation for the `Matrix` type.
    /// Allows modifying elements of the matrix using the `(row, column)` index.
    /// Required so you can mutate an element of the matrix.
    impl<T: Clone> IndexMut<(usize, usize)> for Matrix<T> {
        fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
            assert!(
                index.0 < self.n_rows && index.1 < self.n_columns,
                "Index out of bounds"
            );
            &mut self.data[index.0 * self.n_columns + index.1]
        }
    }

    /// Clone implementation for the `Matrix` type.
    /// Creates a deep copy of the matrix.
    impl<T> Clone for Matrix<T>
    where
        T: Clone,
    {
        fn clone(&self) -> Matrix<T> {
            Matrix {
                n_rows: self.n_rows,
                n_columns: self.n_columns,
                data: self.data.clone(),
            }
        }
    }
    /// helper function to convert a linear index to a matrix index
    /// helpful with scratch regions vs moments matrix
    pub fn linear_to_matrix_index(
        linear_index: usize,
        &(n_cols, _n_rows): &(usize, usize),
    ) -> (usize, usize) {
        (linear_index / n_cols, linear_index % n_cols)
    }
}
