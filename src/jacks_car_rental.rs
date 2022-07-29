use std::collections::HashMap;

use crate::value_iteration::{Probability, ValueIterationTask};

const GAMMA: f64 = 0.9;

pub struct JacksCarRental;
impl ValueIterationTask<State, Action> for JacksCarRental {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn probabilities(&self, s: &State, a: &Action) -> Vec<Probability<State>> {
        let mut probabilities = vec![];

        for in0 in poisson_3() {
            for out0 in poisson_3() {
                for in1 in poisson_2() {
                    for out1 in poisson_4() {
                        let s0_ = (s.0 as i32 - *a) as u32;
                        let s1_ = (s.1 as i32 + *a) as u32;
                        let out0_ = if out0.0 > s0_ { 0 } else { out0.0 };
                        let out1_ = if out1.0 > s1_ { 0 } else { out1.0 };
                        let p = Probability {
                            probability: in0.1 * in1.1 * out0.1 * out1.1,
                            next_state: (
                                u32::min(20, s0_ - out0_ + in0.0),
                                u32::min(20, s1_ - out1_ + in1.0),
                            ),
                            reward: (out0_ + out1_) as f64 * 10.0 + i32::abs(*a) as f64 * -2.0,
                        };
                        probabilities.push(p);
                    }
                }
            }
        }

        // let sum = probabilities.iter().map(|x| x.probability).sum::<f64>();
        // assert!(sum > 0.99);
        probabilities
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        let move_left = u32::min(s.1, 5);
        let move_left = u32::min(20 - s.0, move_left);
        let move_right = u32::min(s.0, 5);
        let move_right = u32::min(20 - s.1, move_right);

        Box::new(-(move_left as i32)..move_right as i32 + 1)
    }

    fn state_space(&self) -> Box<dyn Iterator<Item = State>> {
        let mut states = vec![];
        for i in 0..=20 {
            for j in 0..=20 {
                if i == 0 && j == 0 {
                    continue;
                }
                states.push((i, j));
            }
        }
        Box::new(states.into_iter())
    }

    fn terminal_state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(vec![(0, 0)].into_iter())
    }
}

type State = (u32, u32);
type Action = i32;

// https://ux1.eiu.edu/~aalvarado2/levine-smume6_topic_POIS.pdf
fn poisson_2() -> HashMap<u32, f64> {
    let mut map = HashMap::new();
    map.insert(0, 0.1353);
    map.insert(1, 0.2707);
    map.insert(2, 0.2707);
    map.insert(3, 0.1804);
    map.insert(4, 0.0902);
    map.insert(5, 0.0361);
    map.insert(6, 0.0120);
    // map.insert(7, 0.0034);
    // map.insert(8, 0.0002);
    map
}
fn poisson_3() -> HashMap<u32, f64> {
    let mut map = HashMap::new();
    map.insert(0, 0.0498);
    map.insert(1, 0.1494);
    map.insert(2, 0.2240);
    map.insert(3, 0.2240);
    map.insert(4, 0.1680);
    map.insert(5, 0.1008);
    map.insert(6, 0.0504);
    map.insert(7, 0.0216);
    // map.insert(8, 0.0081);
    // map.insert(9, 0.0027);
    // map.insert(10, 0.0008);
    // map.insert(11, 0.0002);
    // map.insert(12, 0.0001);
    map
}
fn poisson_4() -> HashMap<u32, f64> {
    let mut map = HashMap::new();
    map.insert(0, 0.0183);
    map.insert(1, 0.0733);
    map.insert(2, 0.1465);
    map.insert(3, 0.1954);
    map.insert(4, 0.1954);
    map.insert(5, 0.1563);
    map.insert(6, 0.1042);
    map.insert(7, 0.0595);
    map.insert(8, 0.0298);
    map.insert(9, 0.0132);
    // map.insert(10, 0.0053);
    // map.insert(11, 0.0019);
    // map.insert(12, 0.0006);
    // map.insert(13, 0.0002);
    // map.insert(14, 0.0001);
    map
}
