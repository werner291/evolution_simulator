use id_arena::{Arena, Id};

#[derive(Copy, Clone, Debug)]
pub enum ActivationFunction {
    Linear,
    Tanh,
}

struct Neuron {
    inputs: Vec<(NeuronId, f32)>,
    act_fn: ActivationFunction,
}

type NeuronId = Id<Neuron>;

struct DAGBrain {

    nodes: Arena<Neuron>

}