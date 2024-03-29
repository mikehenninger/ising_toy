I wanted to explore programming in Rust.  As I am wont to do, I used the Ising Model of a lattice of interacting spins as my simulation test bed.  

It ran fine single-threaded, but part of my learning plan was to get a taste of multithreading in Rust, so I made it multithreaded, with the commands as passed messages and data access via shared memory.

I don't claim any of this is done the right way--code or simulation algorithm--but I got a taste of Rust and had some fun! Enjoy.

The code is reasonably well commented and has some rust documentation, so I encourage you to look there.
