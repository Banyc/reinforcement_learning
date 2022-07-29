use std::collections::HashMap;

pub trait ValueIterationTask<State, Action>
where
    State: std::hash::Hash + std::cmp::Eq,
    Action: Copy,
{
    fn gamma(&self) -> f64;
    fn probabilities(&self, s: &State, a: &Action) -> Vec<Probability<State>>;
    fn action_space(&self, s: &State) -> Box<dyn Iterator<Item = Action>>;
    fn state_space(&self) -> Box<dyn Iterator<Item = State>>;
    fn terminal_state_space(&self) -> Box<dyn Iterator<Item = State>>;
}

pub struct ValueIteration<State, Action>
where
    State: std::hash::Hash + std::cmp::Eq,
    Action: Copy,
{
    task: Box<dyn ValueIterationTask<State, Action>>,
}

impl<State, Action> ValueIteration<State, Action>
where
    State: std::hash::Hash + std::cmp::Eq,
    Action: Copy,
{
    pub fn new(task: Box<dyn ValueIterationTask<State, Action>>) -> Self {
        let this = Self { task };
        this
    }
    // type Value = HashMap<State, f64>;
    pub fn value_iteration(&self, theta: f64, v: &mut HashMap<State, f64>) {
        for s in self.task.terminal_state_space() {
            v.insert(s, 0.0);
        }

        let mut delta = f64::MAX;
        while delta >= theta {
            delta = 0.0;
            for s in self.task.state_space() {
                let old_v = v[&s];
                let (new_v, _) = self.max_v_a(&v, &s);
                v.insert(s, new_v);
                delta = f64::max(delta, f64::abs(new_v - old_v));
            }
        }
    }

    pub fn max_v_a(&self, v: &HashMap<State, f64>, s: &State) -> (f64, Vec<Action>) {
        let mut max_v = f64::MIN;
        let mut max_a = vec![];
        for a in self.task.action_space(s) {
            let mut expected_v = 0.0;
            let probabilities = self.task.probabilities(s, &a);
            for probability in probabilities {
                expected_v += probability.probability
                    * (probability.reward + self.task.gamma() * v[&probability.next_state]);
            }
            if max_v < expected_v {
                max_a = vec![a];
            }
            if max_v == expected_v {
                max_a.push(a);
            }
            max_v = f64::max(max_v, expected_v);
        }
        (max_v, max_a)
    }

    pub fn task(&self) -> &Box<dyn ValueIterationTask<State, Action>> {
        &self.task
    }
}

pub struct Probability<State> {
    pub probability: f64,
    pub next_state: State,
    pub reward: f64,
}
