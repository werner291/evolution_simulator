
use std::prelude::v1::Vec;
use rand::Rng;
use ndarray::{Array1, Array2, s, stack, Axis,array};

#[derive(Copy, Clone, Debug)]
pub enum ActivationFunction {
    Linear,
    Tanh,
}


impl ActivationFunction {
    fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Linear => x
        }
    }
}


#[derive(Clone, Debug)]
pub struct Layer {
    weights: Array2<f32>,
    activation: ActivationFunction
}

impl Layer {
    pub fn input_size(&self) -> usize {
        return self.weights.len_of(Axis(1)) - 1; // -1 for bias neuron.
    }

    pub fn output_size(&self) -> usize {
        return self.weights.len_of(Axis(0));
    }
}


#[derive(Clone, Debug)]
pub struct FFNN {
    layers: Vec<Layer>
}

impl FFNN {
    pub fn compute(&self, inputs: Array1<f32>) -> Array1<f32> {
//        eprintln!("Cmp {:?}", self.layers.iter().map(|x| x.weights.raw_dim()).collect::<Vec<_>>());

        debug_assert_eq!(self.input_size(), inputs.len());

        return self.layers.iter()
            .fold(inputs,
                  |i, layer| layer.weights.dot(&stack![Axis(0),i.view(),array![1.0].view()])
                                          .map(|x| layer.activation.apply(*x)));
    }

    pub fn initial(brain_input: usize, brain_output: usize) -> FFNN {
        let mut layers = Vec::new();

        for (i, o, act_fn) in [ (brain_input, 5, ActivationFunction::Tanh)
                              , (5, 5, ActivationFunction::Tanh)
                              , (5, brain_output, ActivationFunction::Linear)].iter() {

            let weights: Array2<f32> = Array2::zeros((*o, *i + 1));

            layers.push(Layer {weights, activation: *act_fn});
        }

        return FFNN {
            layers
        }
    }

    pub fn input_size(&self) -> usize {
        return self.layers[0].input_size();
    }

    pub fn output_size(&self) -> usize {
        return self.layers[self.layers.len()-1].output_size();
    }

    pub fn add_noise(&mut self, noise_factor: f32) {
        let mut rng = rand::thread_rng();

        for l in self.layers.iter_mut() {
            l.weights.mapv_inplace(|x| x + rng.gen_range(-noise_factor, noise_factor));
        }
    }

    pub fn insert_input(&mut self, index: usize) {
        self.insert_node(0, index);
    }

    pub fn insert_output(&mut self, index: usize) {
        self.insert_node(self.layers.len(), index);
    }

    pub fn insert_node(&mut self, layer: usize, index: usize) {
        if let Some(ref mut l) = self.layers.get_mut(layer) {

            l.weights = stack(Axis(1), &[l.weights.slice(s![..,0..index]),
                Array2::zeros([l.weights.len_of(Axis(0)), 1]).view(),
                l.weights.slice(s![.., index..])]).expect("Insertion failed");
        }
        if layer >= 1 {

//            eprintln!("Pre {:?}", self.layers.iter().map(|x| x.weights.raw_dim()).collect::<Vec<_>>());
//            eprintln!("{:?}", layer);
            if let Some(ref mut l) = self.layers.get_mut(layer - 1) {
                l.weights = stack(Axis(0), &[l.weights.slice(s![0..index,..]),
                    Array2::zeros([1, l.weights.len_of(Axis(1))]).view(),
                    l.weights.slice(s![index..,..])]).expect("Insertion failed");
            }
//            eprintln!("Post {:?}", self.layers.iter().map(|x| x.weights.raw_dim()).collect::<Vec<_>>());

        }


    }
}


#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_input_insert() {

        let mut brain = FFNN::initial(2,1);
        assert_eq!(array![0.0], brain.compute(array![0.0,0.0]));

        brain.add_noise(10.0);

        let r1 = brain.compute(array![5.0,10.0]);

        brain.insert_input(1);

        assert_eq!(r1, brain.compute(array![5.0,99.0,10.0]));
    }

    #[test]
    fn test_output_insert() {

        let mut brain = FFNN::initial(2,2);

        brain.add_noise(10.0);

        let r1 = brain.compute(array![5.0,10.0]);

        brain.insert_output(1);

        assert_eq!(array![r1[0], 0.0, r1[1]], brain.compute(array![5.0,10.0]));
    }
}
