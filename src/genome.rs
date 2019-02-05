
use nalgebra    ::{Vector2};
use std::prelude::v1::Vec;
use rand::Rng;
use crate::ffnn::FFNN;

const MUTATION_RATE: f32 = 0.2;

#[derive(Debug)]
pub struct Genome {
    pub nodes: Vec<Vector2<f32>>,
    pub muscles: Vec<(usize, usize)>,
    pub brain: FFNN,
    pub color: [u8; 3],
    pub mutation_rate: f32,
}

impl Genome {
    pub fn initial() -> Genome {
        let nodes = vec![Vector2::new(-1.0, 1.0), Vector2::new( 0.0, 0.0), Vector2::new( 1.0, 0.0)];

        let connections = vec![(0, 1), (1, 2), (2, 0)];

        let brain_input = Genome::brain_input_size(nodes.len(), connections.len());
        let brain_output = Genome::brain_output_size(nodes.len(), connections.len()); // Just the initial muscle thingy

        let mut rng = rand::thread_rng();

        let mut brain = FFNN::initial(brain_input, brain_output);
        brain.add_noise(rng.gen_range(0.5,5.0));

        return Genome {
            nodes,
            muscles: connections,
            brain,
            color: [rng.gen_range(0,255),
                    rng.gen_range(0,255),
                    rng.gen_range(0,255)],
            mutation_rate: rng.gen_range(0.1,2.0)
        };
    }

    fn brain_input_size(anchors: usize, muscles: usize) -> usize {
        return anchors // Food sensors
            + muscles // Muscle tension
            + 1; // Energy level
    }

    fn brain_output_size(anchors: usize, muscles: usize) -> usize {
        return anchors
            + muscles
            + 1; // Reproduction output )
    }

    fn is_connected(&self) -> bool {

        let mut visited = Vec::new();

        visited.push(false);
        for _i in 1..self.nodes.len() {
            visited.push(true);
        }

        loop {
            let mut changed = false;

            for (a,b) in self.muscles.iter() {
                if visited[*a] && !visited[*b] {
                    changed = true;
                    visited[*b] = true;
                }

                if !visited[*a] && visited[*b] {
                    changed = true;
                    visited[*a] = true;
                }
            }

            if !changed {
                break;
            }
        }

        return visited.iter().all(|x| *x);

    }

    pub fn maintenance_price(&self) -> f32 {
        let n_nodes = self.nodes.len();

        let muscle_maintenance = self.muscles.iter().map(|(a,b)| (&self.nodes[*a] - &self.nodes[*b]).norm() ).fold(0.0, |x,y| x+y);

        return n_nodes as f32 + muscle_maintenance;
    }

    fn add_node(&mut self, at: Vector2<f32>) -> usize {

        let nodes_orig = self.nodes.len();

        self.nodes.push(at);
        self.brain.insert_input(nodes_orig);
        self.brain.insert_output(nodes_orig);

        return nodes_orig;
    }

    fn add_connection(&mut self, from: usize, to: usize) -> usize {

        let conns_orig = self.muscles.len();
        self.muscles.push((from, to));
        self.brain.insert_input(self.nodes.len() + conns_orig);
        self.brain.insert_output(self.nodes.len() + conns_orig);
        return conns_orig;
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        const BASE_MUT: f32 = 1.0;
        let mutation_coeff : f32 = BASE_MUT.powf(self.mutation_rate);

        for node in self.nodes.iter_mut() {
            node.x += rng.gen_range(-mutation_coeff, mutation_coeff);
            node.y += rng.gen_range(-mutation_coeff, mutation_coeff);
        }

        self.brain.add_noise(MUTATION_RATE);

//        while false && rng.gen_weighted_bool(5) {
//            let attach_to = rng.gen_range(0, self.nodes.len());
//
//            let new_node = self.add_node(self.nodes[attach_to] + &Vector2::new(
//                rng.gen_range(-mutation_coeff, mutation_coeff),
//                rng.gen_range(-mutation_coeff, mutation_coeff)
//            ));
//
//            self.add_connection(attach_to, new_node);
//        }

        self.color[0] = self.color[0].wrapping_add(rng.gen_range(0,10));
        self.color[1] = self.color[1].wrapping_add(rng.gen_range(0,10));
        self.color[2] = self.color[2].wrapping_add(rng.gen_range(0,10));
        self.mutation_rate += rng.gen_range(-0.1,0.1);

    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use ndarray::prelude::Array1;

    #[test]
    fn test_add() {

        let mut g = Genome::initial();

        for _i in 0..100 {
            g.mutate();
            g.brain.compute(Array1::zeros(g.brain.input_size()));

            assert_eq!(true, g.is_connected());
        }
    }
}