const EAT_RADIUS: f32 = 1.0;
const FOOD_ENERGY: f32 = 10.0;
const MAX_ENERGY: f32 = 100.0;

use crate::sim_framework::Body;
use spade::rtree::RTree;
use nalgebra::{Point2};
use crate::Organism;
use crate::sim_framework::BodyState;

pub struct FoodState {
    pub food: RTree<Point2<f32>>
}

impl FoodState {

    pub fn new() -> FoodState {
        return FoodState {
            food : RTree::new()
        }
    }

    pub fn eat_food(&mut self,
                       organisms: &mut Vec<Organism>,
                       bodies: &mut BodyState)
    {
        for mut org in organisms.iter_mut() {
            let mut body = bodies.get_mut(org.body).expect("Organism should have body.");
            for (n_idx, node) in body.nodes.iter().enumerate() {
                org.food_sensors[n_idx] *= 0.9;

                let eaten: Vec<Point2<f32>> =
                    self.food
                        .lookup_in_circle(&node.pos, &EAT_RADIUS)
                        .iter()
                        .map(|x| Point2::new(x.x, x.y))
                        .collect();

                for i in 0..eaten.len() {
                    self.food.remove(&eaten[i]);
                    body.energy = (body.energy + FOOD_ENERGY).min(MAX_ENERGY);
                    org.food_sensors[n_idx] += 1.0;
                }
            }
        }
    }
}