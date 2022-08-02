use std::collections::HashMap;

use plotly::{common::Mode, Plot, Scatter};
use reinforcement_learning::monte_carlo::{gambler::Gambler, monte_carlo::MonteCarlo};

const EPSILON: f64 = 0.01;
const NUM_EPISODES: usize = 10_000;

fn main() {
    let task = Gambler;
    let monte_carlo = MonteCarlo::new(Box::new(task));
    let mut q = HashMap::new();
    let mut c = HashMap::new();
    let mut pi = HashMap::new();
    monte_carlo.policy_evaluation(&mut q, &mut c, &mut pi, EPSILON, NUM_EPISODES);

    let task = Gambler;
    println!("(s, a)");
    for s in task.state_space() {
        println!("({}, {:?})", s, pi.get(&s));
    }

    {
        let mut plot = Plot::new();
        let mut x = vec![];
        let mut y = vec![];
        for s in task.state_space() {
            if let Some(a) = pi.get(&s) {
                for a in a {
                    x.push(s);
                    y.push(*a);
                }
            }
        }
        let trace = Scatter::new(x, y).name("\\pi(s)").mode(Mode::Markers);
        plot.add_trace(trace);
        plot.show();
    }
}
