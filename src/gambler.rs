use crate::value_iteration::{Probability, ValueIterationTask};

const GAMMA: f64 = 1.0;
const HEAD_PROBABILITY: f64 = 0.4;

pub struct Gambler;
impl ValueIterationTask<State, Action> for Gambler {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn probabilities(&self, s: &State, a: &Action) -> Vec<Probability<State>> {
        if s < a {
            panic!();
        }

        let mut probabilities = vec![];

        {
            let s_ = i32::min(s + a, 100);
            let probability = Probability {
                probability: HEAD_PROBABILITY,
                next_state: s_,
                reward: if s_ == 100 { 1.0 } else { 0.0 },
            };
            probabilities.push(probability);
        }

        {
            let s_ = s - a;
            let probability = Probability {
                probability: 1.0 - HEAD_PROBABILITY,
                next_state: s_,
                reward: 0.0,
            };
            probabilities.push(probability);
        }

        assert!(probabilities.iter().map(|x| x.probability).sum::<f64>() > 0.99);
        probabilities
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        Box::new(0..*s + 1)
    }

    fn state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(1..99 + 1)
    }

    fn terminal_state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(vec![0, 100].into_iter())
    }
}

type State = i32;
type Action = i32;
