use rand::{seq::SliceRandom, Rng};

use crate::{monte_carlo::monte_carlo::MonteCarloTask, q_learning::QLearningTask};

const GAMMA: f64 = 1.0;

pub struct Blackjack;

impl Blackjack {
    fn action_space(&self, _s: &State) -> Box<dyn Iterator<Item = Action>> {
        Box::new(vec![Action::Hit, Action::Stick].into_iter())
    }

    fn action_space_len(&self, _s: &State) -> usize {
        2
    }

    fn random_action(&self, _s: &State) -> Action {
        let mut rng = rand::thread_rng();
        let a = rng.gen_range(0..2);
        match a {
            0 => Action::Hit,
            1 => Action::Stick,
            _ => panic!(),
        }
    }

    fn random_state(&self) -> State {
        let mut rng = rand::thread_rng();
        State {
            dealer: gen_card(),
            me: rng.gen_range(12..21 + 1),
            useful_ace: rng.gen(),
            after_stick: false,
        }
    }

    fn transit(&self, s: &State, a: &Action) -> (State, f64) {
        let mut s_next = *s;
        let card = gen_card();
        let r = match a {
            Action::Hit => s_next.me_get_card(card),
            Action::Stick => s_next.dealer_get_card(card),
        };
        (s_next, r)
    }

    fn in_terminal_state_space(&self, s: &State) -> bool {
        if s.after_stick {
            return true;
        }
        if s.me_busted() {
            return true;
        }
        false
    }
}
impl MonteCarloTask<State, Action> for Blackjack {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        self.action_space(s)
    }

    fn action_space_len(&self, s: &State) -> usize {
        self.action_space_len(s)
    }

    fn random_action(&self, s: &State) -> Action {
        self.random_action(s)
    }

    fn random_state(&self) -> State {
        self.random_state()
    }

    fn transit(&self, s: &State, a: &Action) -> (State, f64) {
        self.transit(s, a)
    }

    fn in_terminal_state_space(&self, s: &State) -> bool {
        self.in_terminal_state_space(s)
    }
}
impl QLearningTask<State, Action> for Blackjack {
    fn gamma(&self) -> f64 {
        GAMMA
    }

    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>> {
        self.action_space(s)
    }

    fn action_space_len(&self, s: &State) -> usize {
        self.action_space_len(s)
    }

    fn random_action(&self, s: &State) -> Action {
        self.random_action(s)
    }

    fn random_state(&self) -> State {
        self.random_state()
    }

    fn transit(&self, s: &State, a: &Action) -> (State, f64) {
        self.transit(s, a)
    }

    fn in_terminal_state_space(&self, s: &State) -> bool {
        self.in_terminal_state_space(s)
    }
}

fn gen_card() -> u32 {
    *[2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        .choose(&mut rand::thread_rng())
        .unwrap()
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct State {
    pub dealer: u32,
    pub me: u32,
    pub useful_ace: bool,
    pub after_stick: bool,
}
impl State {
    pub fn me_busted(&self) -> bool {
        21 < self.me
    }
    pub fn dealer_busted(&self) -> bool {
        21 < self.dealer
    }

    pub fn me_get_card(&mut self, card: u32) -> f64 {
        self.me += card;
        if card == 11 {
            self.useful_ace = true;
        }
        // save me by ace
        if self.me_busted() && self.useful_ace {
            self.me -= 10;
            self.useful_ace = false;
        }
        if self.me_busted() {
            return -1.;
        }
        0.
    }

    pub fn dealer_get_card(&mut self, card: u32) -> f64 {
        self.after_stick = true;
        let prev = self.dealer;
        self.dealer += card;
        let useful_ace = card == 11 || prev == 11;
        // save dealer by ace
        if self.dealer_busted() && useful_ace {
            self.dealer -= 10;
        }
        if self.dealer_busted() {
            return 1.0;
        }
        match self.dealer.cmp(&self.me) {
            std::cmp::Ordering::Less => 1.,
            std::cmp::Ordering::Equal => 0.,
            std::cmp::Ordering::Greater => -1.,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum Action {
    Hit,
    Stick,
}
