use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path,
};

use reinforcement_learning::value_iteration::{
    jacks_car_rental::JacksCarRental, value_iteration::ValueIteration,
};

const VALUE_OUTPUT_FILE: &str = "jacks_car_rental.value.csv";
const ACTION_OUTPUT_FILE: &str = "jacks_car_rental.action.csv";

fn main() {
    let task = JacksCarRental;
    let value_iteration = ValueIteration::new(Box::new(task));
    let mut v = HashMap::new();
    for s in value_iteration.task().state_space() {
        v.insert(s, 0.0);
    }
    value_iteration.value_iteration(10.0, &mut v);
    {
        println!("(s, V(s))");
        if let Err(e) = fs::remove_file(path::Path::new(VALUE_OUTPUT_FILE)) {
            if e.kind() != std::io::ErrorKind::NotFound {
                panic!("{}", e);
            }
        }
        let mut file = File::create(VALUE_OUTPUT_FILE).unwrap();
        for s in value_iteration.task().state_space() {
            println!("({:?}, {})", s, v[&s]);
            writeln!(file, "{}, {}, {}", s.0, s.1, v[&s]).unwrap();
        }
    }
    println!();
    {
        println!("(s, a)");
        if let Err(e) = fs::remove_file(path::Path::new(ACTION_OUTPUT_FILE)) {
            if e.kind() != std::io::ErrorKind::NotFound {
                panic!("{}", e);
            }
        }
        let mut file = File::create(ACTION_OUTPUT_FILE).unwrap();
        for s in value_iteration.task().state_space() {
            let a = value_iteration.max_v_a(&v, &s).1;
            println!("({:?}, {:?})", s, a);
            for a in a {
                writeln!(file, "{}, {}, {}", s.0, s.1, a).unwrap();
            }
        }
    }
}
