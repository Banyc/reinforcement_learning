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

    /// - `value`: $Q$
    /// - `weight_sum`: $C$
    /// - `policy`: $\pi$
    /// - `prob_explore`: $\epsilon$
    pub fn policy_evaluation(
        &self,
        value: &mut HashMap<StateActionPair<State, Action>, f64>,
        l_importance_sum: &mut HashMap<StateActionPair<State, Action>, f64>,
        policy: &mut HashMap<State, Vec<Action>>,
        prob_explore: f64,
        num_episodes: usize,
    ) {
        for _ in 0..num_episodes {
            let episode = self.generate_episode(policy, prob_explore);

            let mut step_ret = 0.0;
            let mut learning_importance = 1.0;
            for step in episode.iter().rev() {
                step_ret = self.task.gamma() * step_ret + step.reward;
                let state_then_action = StateActionPair {
                    state: step.state,
                    action: step.action,
                };
                let learning_rate = {
                    // learning rate decays in an $1/n$ manner on $n$-th step
                    let l_importance_sum = l_importance_sum.entry(state_then_action).or_insert(0.);
                    *l_importance_sum += learning_importance;
                    learning_importance / *l_importance_sum
                };
                {
                    // Nudge value towards the step return for this action on the current state
                    let v = value.get(&state_then_action).unwrap_or(&0.0);
                    let shift = learning_rate * (step_ret - v);
                    value.insert(state_then_action, v + shift);
                }
                let best_actions = {
                    // Set the best actions to the policy
                    let (_, a) = self.max_value_by_actions(value, &step.state);
                    policy.insert(step.state, a);
                    policy.get(&step.state).unwrap()
                };
                if !best_actions.contains(&step.action) {
                    // Early terminate the episode that goes on the suboptimal direction
                    break;
                }
                {
                    // Adjust learning rate
                    let prob_explored_best_actions = {
                        let num_best = best_actions.len();
                        let num_all = self.task.action_space_len(&step.state);
                        prob_explore * num_best as f64 / num_all as f64
                    };
                    // Learn more from the immediate past with a correctly explored future
                    learning_importance *= 1. / prob_explored_best_actions;
                }
            }
        }
    }

    fn generate_episode(
        &self,
        policy: &HashMap<State, Vec<Action>>,
        prob_explore: f64,
    ) -> Vec<Step<State, Action>> {
        // using epsilon-greedy policy b
        let mut episode = vec![];
        let mut s = self.task.random_state();
        while !self.task.in_terminal_state_space(&s) {
            let mut rng = rand::thread_rng();
            let rnd = rng.gen_range(0.0..1.0);
            // epsilon-greedy policy
            let a = if rnd < prob_explore {
                self.task.random_action(&s)
            } else {
                match policy.get(&s) {
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
        episode
    }

    pub fn max_value_by_actions(
        &self,
        value: &HashMap<StateActionPair<State, Action>, f64>,
        s: &State,
    ) -> (f64, Vec<Action>) {
        let mut max_v = f64::MIN;
        let mut max_a = vec![];
        for a in self.task.action_space(s) {
            let v = *value
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
