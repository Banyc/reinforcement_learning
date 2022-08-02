use rand::Rng;

use super::monte_carlo::MonteCarloTask;

const GAMMA: f64 = 1.0;
const HEAD_PROBABILITY: f64 = 0.4;

pub struct Gambler;
impl MonteCarloTask<State, Action> for Gambler {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        Box::new(0..*s + 1)
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
        vec![0, 100].contains(s)
    }
}

impl Gambler {
    pub fn state_space(&self) -> Box<dyn Iterator<Item = State>> {
        Box::new(1..99 + 1)
    }
}

type State = i32;
type Action = i32;
