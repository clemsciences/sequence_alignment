extern crate nalgebra as na;

use na::{Vector3};


fn main() {
    // let v = Vector::from_data([1, 2, 3]);
    let v = Vector3::new(1, 2, 3);
    println!("{}", &v.abs());
    println!("{}", &v.abs());
    println!("{}", &v.abs());
    println!("Hello, world!");
}
