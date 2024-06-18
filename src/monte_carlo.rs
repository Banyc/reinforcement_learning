use std::collections::HashMap;

use rand::{seq::SliceRandom, Rng};

pub trait MonteCarloTask<State, Action> {
    fn gamma(&self) -> f64;
    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>>;
    fn action_space_len(&self, s: &State) -> usize;
    fn random_action(&self, s: &State) -> Action;
    fn random_state(&self) -> State;
    fn transit(&self, s: &State, a: &Action) -> (State, f64);
    fn in_terminal_state_space(&self, s: &State) -> bool;
}

pub struct MonteCarlo<State, Action> {
    task: Box<dyn MonteCarloTask<State, Action>>,
}

impl<State, Action> MonteCarlo<State, Action>
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    pub fn new(task: Box<dyn MonteCarloTask<State, Action>>) -> Self {
        Self { task }
    }

    pub fn policy_evaluation(
        &self,
        q: &mut HashMap<StateActionPair<State, Action>, f64>,
        c: &mut HashMap<StateActionPair<State, Action>, f64>,
        pi: &mut HashMap<State, Vec<Action>>,
        e: f64,
        num_episodes: usize,
    ) {
        for _ in 0..num_episodes {
            // generate an episode
            // using epsilon-greedy policy b
            let mut episode = vec![];
            {
                let mut s = self.task.random_state();
                while !self.task.in_terminal_state_space(&s) {
                    let mut rng = rand::thread_rng();
                    let rnd = rng.gen_range(0.0..1.0);
                    // epsilon-greedy policy
                    let a = if rnd < e {
                        self.task.random_action(&s)
                    } else {
                        match pi.get(&s) {
                            Some(a) => *a.choose(&mut rng).unwrap(),
                            None => self.task.random_action(&s),
                        }
                    };
                    let (s_next, r) = self.task.transit(&s, &a);
                    episode.push(Step {
                        state: s,
                        action: a,
                        reward: r,
                    });
                    s = s_next;
                }
            }

            let mut g = 0.0;
            let mut w = 1.0;
            for step in episode.iter().rev() {
                g = self.task.gamma() * g + step.reward;
                let s_a = StateActionPair {
                    state: step.state,
                    action: step.action,
                };
                {
                    let c_old = c.get(&s_a).unwrap_or(&0.0);
                    c.insert(s_a, c_old + w);
                }
                {
                    let q_old = q.get(&s_a).unwrap_or(&0.0);
                    let shift = w / c.get(&s_a).unwrap() * (g - q_old);
                    q.insert(s_a, q_old + shift);
                }
                {
                    let (_, a) = self.max_q_a(q, &step.state);
                    pi.insert(step.state, a);
                }
                if !pi.get(&step.state).unwrap().contains(&step.action) {
                    break;
                }
                {
                    let pi_a_len = pi.get(&step.state).unwrap().len();
                    // probability of the action under the epsilon-greedy policy b
                    let b_p_a = {
                        (1.0 - e) / pi_a_len as f64
                            + e / self.task.action_space_len(&step.state) as f64
                    };

                    w *= 1.0 / (b_p_a * pi_a_len as f64);
                }
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

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct StateActionPair<State, Action>
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    pub state: State,
    pub action: Action,
}

struct Step<State, Action> {
    pub state: State,
    pub action: Action,
    pub reward: f64,
}
