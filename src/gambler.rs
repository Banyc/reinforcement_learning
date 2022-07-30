use crate::value_iteration::{Possibility, ValueIterationTask};

const GAMMA: f64 = 1.0;
const HEAD_PROBABILITY: f64 = 0.4;

pub struct Gambler;
impl ValueIterationTask<State, Action> for Gambler {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn possibilities(&self, s: &State, a: &Action) -> Vec<Possibility<State>> {
        if s < a {
            panic!();
        }

        let mut possibilities = vec![];

        {
            let s_ = i32::min(s + a, 100);
            let possibility = Possibility {
                probability: HEAD_PROBABILITY,
                next_state: s_,
                reward: if s_ == 100 { 1.0 } else { 0.0 },
            };
            possibilities.push(possibility);
        }

        {
            let s_ = s - a;
            let probability = Possibility {
                probability: 1.0 - HEAD_PROBABILITY,
                next_state: s_,
                reward: 0.0,
            };
            possibilities.push(probability);
        }

        assert!(possibilities.iter().map(|x| x.probability).sum::<f64>() > 0.99);
        possibilities
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
