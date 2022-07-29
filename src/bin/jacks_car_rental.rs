use std::collections::HashMap;

use value_iteration::{jacks_car_rental::JacksCarRental, value_iteration::ValueIteration};

fn main() {
    let task = JacksCarRental;
    let value_iteration = ValueIteration::new(Box::new(task));
    let mut v = HashMap::new();
    for s in value_iteration.task().state_space() {
        v.insert(s, 0.0);
    }
    value_iteration.value_iteration(10.0, &mut v);
    println!("(s, V(s))");
    for s in value_iteration.task().state_space() {
        println!("({:?}, {})", s, v[&s]);
    }
    println!();
    println!("(s, a)");
    for s in value_iteration.task().state_space() {
        println!("({:?}, {:?})", s, value_iteration.max_v_a(&v, &s).1);
    }
}
