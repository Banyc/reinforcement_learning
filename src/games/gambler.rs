use rand::Rng;

use crate::{
    monte_carlo::MonteCarloTask,
    value_iteration::{Possibility, ValueIterationTask},
};

const GAMMA: f64 = 1.0;
const HEAD_PROBABILITY: f64 = 0.4;

pub struct Gambler;
impl Gambler {
    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        Box::new(0..*s + 1)
    }

    pub fn state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(1..99 + 1)
    }
}
impl MonteCarloTask<State, Action> for Gambler {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        self.action_space(s)
    }

    fn action_space_len(&self, s: &State) -> usize {
        (*s + 1) as usize
    }

    fn random_action(&self, s: &State) -> Action {
        rand::thread_rng().gen_range(0..*s + 1)
    }

    fn random_state(&self) -> State {
        rand::thread_rng().gen_range(1..99 + 1)
    }

    fn transit(&self, s: &State, a: &Action) -> (State, f64) {
        let rnd = rand::thread_rng().gen_range(0.0..1.0);
        if rnd < HEAD_PROBABILITY {
            let s_ = i32::min(s + a, 100);
            let r = if s_ == 100 { 1.0 } else { 0.0 };
            (s_, r)
        } else {
            let s_ = s - a;
            (s_, 0.0)
        }
    }

    fn in_terminal_state_space(&self, s: &State) -> bool {
        [0, 100].contains(s)
    }
}
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
        self.action_space(s)
    }

    fn state_space(&self) -> Box<dyn Iterator<Item = State>> {
        self.state_space()
    }

    fn terminal_state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(vec![0, 100].into_iter())
    }
}

type State = i32;
type Action = i32;
