use rand::{seq::SliceRandom, Rng};

use super::q_learning::QLearningTask;

const GAMMA: f64 = 1.0;

pub struct Blackjack;

impl QLearningTask<State, Action> for Blackjack {
    fn gamma(&self) -> f64 {
        GAMMA
    }

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
        let mut s_next = (*s).clone();
        let mut r = 0.0;
        match a {
            Action::Hit => {
                let card = gen_card();
                s_next.me += card;
                if s_next.me > 21 {
                    if s_next.useful_ace {
                        s_next.me -= 10;
                        if card != 11 {
                            s_next.useful_ace = false;
                        }
                    } else {
                        r = -1.0;
                    }
                }
            }
            Action::Stick => {
                s_next.after_stick = true;
                let card = gen_card();
                s_next.dealer += card;
                if s_next.dealer > 21 {
                    if card == 11 || s.dealer == 11 {
                        s_next.dealer -= 10;
                    }
                }
                if s_next.dealer > 21 {
                    r = 1.0;
                } else if s_next.dealer > s_next.me {
                    r = -1.0;
                } else if s_next.dealer == s_next.me {
                    r = 0.0;
                } else {
                    r = 1.0;
                }
            }
        }
        (s_next, r)
    }

    fn in_terminal_state_space(&self, s: &State) -> bool {
        if s.after_stick {
            true
        } else if s.me > 21 {
            true
        } else {
            false
        }
    }
}

fn gen_card() -> u32 {
    *vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
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

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum Action {
    Hit,
    Stick,
}
