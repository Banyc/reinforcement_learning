use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path,
};

use reinforcement_learning::{games::blackjack::Blackjack, monte_carlo::monte_carlo::MonteCarlo};

const EPSILON: f64 = 0.1;
const NUM_EPISODES: usize = 10_000;
const ACTION_OUTPUT_FILE: &str = "blackjack.action.txt";

fn main() {
    let task = Blackjack;
    let monte_carlo = MonteCarlo::new(Box::new(task));
    let mut q = HashMap::new();
    let mut c = HashMap::new();
    let mut pi = HashMap::new();
    monte_carlo.policy_evaluation(&mut q, &mut c, &mut pi, EPSILON, NUM_EPISODES);

    println!("(s, a)");
    if let Err(e) = fs::remove_file(path::Path::new(ACTION_OUTPUT_FILE)) {
        if e.kind() != std::io::ErrorKind::NotFound {
            panic!("{}", e);
        }
    }
    let mut file = File::create(ACTION_OUTPUT_FILE).unwrap();
    // println!("{:#?}", pi);
    writeln!(file, "{:#?}", pi).unwrap();
}
