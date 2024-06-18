use std::collections::HashMap;

use rand::{seq::SliceRandom, Rng};

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
        q: &mut HashMap<StateActionPair<State, Action>, f64>,
        e: f64,
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
                let a = if rnd < e {
                    self.task.random_action(&s)
                } else {
                    let (_, a) = self.max_q_a(q, &s);
                    *a.choose(&mut rng).unwrap()
                };
                let (s_next, r) = self.task.transit(&s, &a);
                // update Q(S, A)
                {
                    let sa = StateActionPair {
                        state: s,
                        action: a,
                    };
                    let q_sa = *q.get(&sa).unwrap_or(&0.0);
                    let (next_max_q, _) = self.max_q_a(q, &s_next);
                    let new_q_sa = q_sa + alpha * (r + self.task.gamma() * next_max_q - q_sa);
                    q.insert(sa, new_q_sa);
                }

                s = s_next;
            }
        }
    }

    pub fn max_q_a(
        &self,
        q: &HashMap<StateActionPair<State, Action>, f64>,
        s: &State,
    ) -> (f64, Vec<Action>) {
        let mut max_v = f64::MIN;
        let mut max_a = vec![];
        for a in self.task.action_space(s) {
            let v = *q
                .get(&StateActionPair {
                    state: *s,
                    action: a,
                })
                .unwrap_or(&0.0);
            if max_v < v {
                max_a = vec![a];
            }
            if max_v == v {
                max_a.push(a);
            }
            max_v = f64::max(max_v, v);
        }
        (max_v, max_a)
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct StateActionPair<State, Action>
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    pub state: State,
    pub action: Action,
}
