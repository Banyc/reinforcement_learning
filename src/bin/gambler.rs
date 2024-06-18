use std::collections::HashMap;

use plotly::{common::Mode, Plot, Scatter};
use reinforcement_learning::{games::gambler::Gambler, value_iteration::ValueIteration};

fn main() {
    let task = Gambler;
    let value_iteration = ValueIteration::new(Box::new(task));
    let mut v = HashMap::new();
    for s in value_iteration.task().state_space() {
        v.insert(s, 0.0);
    }
    value_iteration.value_iteration(0.01, &mut v);
    println!("(s, V(s))");
    for s in value_iteration.task().state_space() {
        println!("({}, {})", s, v[&s]);
    }
    println!();
    println!("(s, a)");
    for s in value_iteration.task().state_space() {
        println!("({}, {:?})", s, value_iteration.max_v_a(&v, &s).1);
    }

    {
        let mut plot = Plot::new();
        let mut x = vec![];
        let mut y = vec![];
        for s in value_iteration.task().state_space() {
            x.push(s);
            y.push(v[&s]);
        }
        let trace = Scatter::new(x, y).name("V(s)").mode(Mode::Lines);
        plot.add_trace(trace);
        plot.show();
    }
    {
        let mut plot = Plot::new();
        let mut x = vec![];
        let mut y = vec![];
        for s in value_iteration.task().state_space() {
            let a = value_iteration.max_v_a(&v, &s).1;
            for a in a {
                x.push(s);
                y.push(a);
            }
        }
        let trace = Scatter::new(x, y).name("\\pi(s)").mode(Mode::Markers);
        plot.add_trace(trace);
        plot.show();
    }
}
