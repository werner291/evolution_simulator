const EAT_RADIUS: f32 = 1.0;
const FOOD_ENERGY: f32 = 10.0;
const MAX_ENERGY: f32 = 100.0;

use crate::body_sim::Body;
use spade::rtree::RTree;
use nalgebra::{Point2};

pub struct FoodState {
    pub food: RTree<Point2<f32>>
}

impl FoodState {
    pub fn eat_food<T>(&mut self, bodies: &mut T)
        where T: Iterator<Item=Body>
    {
        for mut body in bodies.iter_mut() {
            for node in body.nodes.iter() {
                node.food_sensor *= 0.9;

                let eaten: Vec<Point2<f32>> =
                    self.food
                        .lookup_in_circle(&node.pos, &EAT_RADIUS)
                        .iter()
                        .map(|x| Point2::new(x.x, x.y))
                        .collect();

                for i in 0..eaten.len() {
                    self.food.remove(&eaten[i]);
                    body.energy = (body.energy + FOOD_ENERGY).min(MAX_ENERGY);
                    node.food_sensor += 1.0;
                }
            }
        }
    }
}