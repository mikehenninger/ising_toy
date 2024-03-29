use crate::ultralight_matrix::matrix::*;
use core::panic;
use rand::Rng;
use std::sync::{mpsc, Arc, Barrier, Mutex, RwLock};
use std::thread;

/// Represents an update worker for the simulation.
/// Each worker runs in its own thread and updates a portion of the lattice.
/// The worker takes read access on the RwLocks for input matrices and mutex
/// lock to update its scratch region--which maps to moments in the main lattice.
/// The worker gets UpdateMessages from the parent Lattice, and the parent also
/// copies the updated moments from the scratch region back to the main `moments`.
pub struct UpdateWorker<F: Clone + Send> {
    pub id: usize,
    pub thread: Option<thread::JoinHandle<()>>,
    pub moments: Arc<RwLock<Matrix<f64>>>,
    pub temperature: Arc<RwLock<Matrix<f64>>>,
    pub external_field: Arc<RwLock<Matrix<f64>>>,
    pub scratch_region: Arc<Mutex<Vec<f64>>>,
    pub scratch_offset: usize,
    scratch_len: usize,
    pub sender: mpsc::SyncSender<UpdateMessage>,
    hamiltonian: F,
}
impl<F: Clone + Send> UpdateWorker<F> {
    /// This is enormous and I need to set aside some time to decompose it.
    /// The magic happens in the giant closure run by the spawned thread.
    /// The struct gets both ends of a channel, but `UpdateWorker.sender` isn't
    /// moved into the thread and it's where you submit your message.  The channel
    /// is SPSC, with each thread having its own.
    /// The closure is a loop that waits for a message, then updates the scratch region
    /// there is a barrier at the end of the loop that ensures all threads have finished
    /// the last barrier member is in `Lattice.copy_from_scratch``, which holds
    /// on copying scratch to `moments` until all threads have finished updating.
    /// The barrier waits at the END of each thread's loop, but the START of
    /// `Lattice.copy_from_scratch`

    pub fn new(
        id: usize,
        sender: mpsc::SyncSender<UpdateMessage>, //NOT moved into the thread
        receiver: mpsc::Receiver<UpdateMessage>,
        scratch_barrier: Arc<Barrier>,
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
        let scratch_len = scratch_region.lock().unwrap().len();
        let thread = thread::Builder::new()
            .spawn(move || loop {
                let update_message = match receiver.recv() {
                    Ok(update_message) => update_message,
                    Err(_) => break,
                };
                // XXX TODO: try removing the barrier and moving the scratch mutex lock up here
                // _should_ do the same thing, right?  or is there a chance the main thread
                // will start copy_from_scratch before this thread even parses the message?
                // cant lock before the message or it'd be perma-locked
                let read_moments = moments.read().expect("Could not read moments");
                let read_external_field = external_field
                    .read()
                    .expect("Could not read external field");

                let update_loc = match update_message {
                    UpdateMessage::Location(i, j) => vec![(i as i64, j as i64)],
                    UpdateMessage::All => {
                        let mut all_locs = Vec::new();
                        for idx_absolute in
                            scratch_offset..(scratch_offset + scratch_region.lock().unwrap().len())
                        {
                            all_locs.push((
                                (idx_absolute / read_moments.n_columns) as i64,
                                (idx_absolute % read_moments.n_columns) as i64,
                            ));
                        }
                        all_locs
                    }
                    UpdateMessage::Stop => break, //kill the spawn loop and thus the thread
                    UpdateMessage::Ignore => {
                        //don't want this thread to do anything EXCEPT do its part to trigger the barrier
                        scratch_barrier.wait();
                        continue;
                    }
                    UpdateMessage::UpdateN(n) => {
                        let mut rng = rand::thread_rng();
                        let mut all_locs = Vec::new();

                        for _i in 0..n {
                            let local_offset = rng.gen_range(0..scratch_len);

                            let (ui, uj) = linear_to_matrix_index(
                                local_offset + scratch_offset,
                                &(read_moments.n_columns, read_moments.n_rows),
                            );
                            all_locs.push((ui as i64, uj as i64));
                        }
                        all_locs
                    }
                };

                {
                    // mutex locked scope
                    let mut scratch_region_locked = scratch_region
                        .lock()
                        .expect(&format!("couldn't lock scratch region for thread {id}"));
                    for (i, j) in update_loc {
                        let (ui, uj) = read_moments.wrap((i, j));

                        let offset_linear_loc = ui * read_moments.n_columns + uj - scratch_offset;

                        let e_site = hamiltonian(&read_moments, &read_external_field, &(i, j));
                        let t_site = temperature.read().unwrap()[(ui, uj)];
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

                scratch_barrier.wait();
            })
            .expect("Could not create thread");
        UpdateWorker {
            id,
            sender,
            thread: Some(thread),
            moments: ref_copy_moments,
            temperature: ref_copy_temperature,
            external_field: ref_copy_external_field,
            hamiltonian: ref_copy_hamiltonian,
            scratch_region: ref_copy_scratch_region,
            scratch_offset, //survives all the way to here via copy
            scratch_len,
        }
    }
}

/// The possible messages that can be sent to an update worker.
/// `Location(i, j)` indicates that the worker should update the site at row `i` and column `j`.
/// `All` indicates that the worker should update all sites in its scratch region.
/// `Ignore` indicates that the worker should ignore the message, but still trigger the barrier.
/// `Stop` indicates that the worker should stop processing messages and exit.
/// `UpdateN(n)` indicates that the worker should update `n` random sites in its scratch region.
pub enum UpdateMessage {
    Location(usize, usize),
    All,
    Ignore,
    Stop,
    UpdateN(usize),
}
