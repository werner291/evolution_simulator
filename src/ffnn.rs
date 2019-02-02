
extern crate sdl2;
extern crate chrono;
extern crate nalgebra as na;
extern crate rand;
extern crate ndarray as nd;

use std::prelude::v1::Vec;
use rand::Rng;
use nd::{Array1, Array2, s};
use ndarray::Axis;

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
        return self.weights.len_of(Axis(1));
    }

    pub fn output_size(&self) -> usize {
        return self.weights.len_of(Axis(0));
    }

    pub fn expanded_copy(&self, inpt: usize, oupt: usize) -> Layer {
        let mut new_weights: Array2<f32> = Array2::zeros((
            oupt, inpt
        ));

        new_weights.slice_mut(s![ ..self.output_size()
                                , ..self.input_size() ]).assign(&self.weights);

        return Layer {
            weights: new_weights,
            activation: self.activation
        }
    }
}

#[derive(Clone, Debug)]
pub struct FFNN {
    layers: Vec<Layer>
}

impl FFNN {
    pub fn compute(&self, inputs: Array1<f32>) -> Array1<f32> {
        return self.layers.iter().fold(inputs, |i, layer| layer.weights.dot(&i).map(|x| layer.activation.apply(*x)));
    }

    pub fn initial(brain_input: usize, brain_output: usize) -> FFNN {
        let mut layers = Vec::new();

        for (i, o, act_fn) in [ (brain_input, 5, ActivationFunction::Tanh)
                              , (5, 5, ActivationFunction::Tanh)
                              , (5, brain_output, ActivationFunction::Linear)].iter() {

            let mut layer: Array2<f32> = Array2::zeros((*o, *i));

            let mut rng = rand::thread_rng();

            layer.mapv_inplace(|x| rng.gen_range(-1.0, 1.0));

            layers.push(Layer {weights: layer, activation: *act_fn});
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

    pub fn mutated_copy(&self, mutation_rate: f32, brain_input: usize, brain_output: usize) -> FFNN {

        return FFNN {
            layers: (0..self.layers.len()).map(|i| {
                let mut l =
                    if i == 0 {
                        self.layers[i].expanded_copy(brain_input, self.layers[i].output_size())
                    } else if i == self.layers.len() - 1 {
                        self.layers[i].expanded_copy(self.layers[i].input_size(), brain_output)
                    } else {
                        self.layers[i].clone()
                    };

                let mut rng = rand::thread_rng();

                l.weights.mapv_inplace(|x| x + rng.gen_range(-mutation_rate, mutation_rate));

                return l

            }).collect()
        }
    }
}