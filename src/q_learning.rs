use std::collections::HashMap;

use rand::{seq::SliceRandom, Rng};

use crate::{max_value_by_actions, StateActionPair};

pub trait QLearningTask<State, Action> {
    fn gamma(&self) -> f64;
    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>>;
    fn action_space_len(&self, s: &State) -> usize;
    fn random_action(&self, s: &State) -> Action;
    fn random_state(&self) -> State;
    fn transit(&self, s: &State, a: &Action) -> (State, f64);
    fn in_terminal_state_space(&self, s: &State) -> bool;
}

pub struct QLearning<State, Action> {
    task: Box<dyn QLearningTask<State, Action>>,
}

impl<State, Action> QLearning<State, Action>
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    pub fn new(task: Box<dyn QLearningTask<State, Action>>) -> Self {
        Self { task }
    }

    pub fn value_evaluation(
        &self,
        value: &mut HashMap<StateActionPair<State, Action>, f64>,
        prob_explore: f64,
        alpha: f64,
        num_episodes: usize,
    ) {
        for _ in 0..num_episodes {
            // using epsilon-greedy policy b
            let mut s = self.task.random_state();
            while !self.task.in_terminal_state_space(&s) {
                let mut rng = rand::thread_rng();
                let rnd = rng.gen_range(0.0..1.0);
                // epsilon-greedy policy
                let a = if rnd < prob_explore {
                    self.task.random_action(&s)
                } else {
                    let (_, a) = self.max_value_by_actions(value, &s);
                    *a.choose(&mut rng).unwrap()
                };
                let (s_next, r) = self.task.transit(&s, &a);
                // update Q(S, A)
                {
                    let state_then_action = StateActionPair {
                        state: s,
                        action: a,
                    };
                    let q_sa = *value.get(&state_then_action).unwrap_or(&0.0);
                    let (next_max_value, _) = self.max_value_by_actions(value, &s_next);
                    let new_q_sa = q_sa + alpha * (r + self.task.gamma() * next_max_value - q_sa);
                    value.insert(state_then_action, new_q_sa);
                }

                s = s_next;
            }
        }
    }

    pub fn max_value_by_actions(
        &self,
        value: &HashMap<StateActionPair<State, Action>, f64>,
        s: &State,
    ) -> (f64, Vec<Action>) {
        max_value_by_actions(value, s, self.task.action_space(s))
    }
}
