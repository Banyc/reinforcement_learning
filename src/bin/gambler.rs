use value_iteration::{gambler::Gambler, value_iteration::ValueIteration};

fn main() {
    let task = Gambler;
    let value_iteration = ValueIteration::new(Box::new(task));
    let v = value_iteration.value_iteration(0.01);
    println!("(s, V(s))");
    for s in value_iteration.task().state_space() {
        println!("({}, {})", s, v[&s]);
    }
    println!();
    println!("(s, a)");
    for s in value_iteration.task().state_space() {
        println!("({}, {:?})", s, value_iteration.max_v_a(&v, &s).1);
    }
}
