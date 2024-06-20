use std::collections::HashMap;

pub mod games;
pub mod monte_carlo;
pub mod q_learning;
pub mod value_iteration;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct StateActionPair<State, Action>
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    pub state: State,
    pub action: Action,
}

pub fn max_value_by_actions<State, Action>(
    value: &HashMap<StateActionPair<State, Action>, f64>,
    s: &State,
    action_space: impl Iterator<Item = Action>,
) -> (f64, Vec<Action>)
where
    State: Copy + std::hash::Hash + std::cmp::Eq,
    Action: Copy + std::hash::Hash + std::cmp::Eq,
{
    let mut max_v = f64::MIN;
    let mut max_a = vec![];
    for a in action_space {
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
