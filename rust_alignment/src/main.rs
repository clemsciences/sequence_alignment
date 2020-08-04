extern crate nalgebra as na;

use na::Vector;
// use nalgebra;


fn main() {
    // let v = Vector::from_data([1, 2, 3]);
    let v = Vector3::new(1, 2, 3);
    println!("{}", &v.abs());
    println!("{}", &v.abs());
    println!("{}", &v.abs());
    println!("Hello, world!");
}
