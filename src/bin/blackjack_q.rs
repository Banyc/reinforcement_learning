use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path,
};

use reinforcement_learning::{games::blackjack::Blackjack, q_learning::QLearning};

const EPSILON: f64 = 0.1;
const ALPHA: f64 = 0.1;
const NUM_EPISODES: usize = 10_000;
const VALUE_OUTPUT_FILE: &str = "blackjack.Q_learning.action_value.txt";

fn main() {
    let task = Blackjack;
    let q_learning = QLearning::new(Box::new(task));
    let mut q = HashMap::new();
    q_learning.value_evaluation(&mut q, EPSILON, ALPHA, NUM_EPISODES);

    println!("((s, a), Q(s, a))");
    if let Err(e) = fs::remove_file(path::Path::new(VALUE_OUTPUT_FILE)) {
        if e.kind() != std::io::ErrorKind::NotFound {
            panic!("{}", e);
        }
    }
    let mut file = File::create(VALUE_OUTPUT_FILE).unwrap();
    // println!("{:#?}", pi);
    writeln!(file, "{:#?}", q).unwrap();
}
