use generational_arena::{Index,Arena};
use nalgebra::{Point2, Vector2};
use crate::genome::Genome;
use nalgebra::distance;
use ndarray::prelude::Array1;

pub const START_ENERGY: f32 = 10.0;
pub const MOVE_ENERGY : f32 = 0.01;
pub const ANCHOR_STRENGTH : f32 = 1.0;

pub struct Node {
    pub pos: Point2<f32>,
    pub vel: Vector2<f32>,
    pub anchor: f32,
}

pub struct Muscle {
    pub from: usize,
    pub to: usize,
    pub natural_length: f32,
    pub extension_factor: f32,
}

pub struct Body {
    pub nodes: Vec<Node>,
    pub muscles: Vec<Muscle>,
    pub energy: f32
}

impl Body {
    pub fn center(&self) -> Point2<f32> {
        return self.nodes.iter().fold(Point2::new(0.0, 0.0),
                                      |p1, p2| Point2::new(p1.x + p2.pos.x, p1.y + p2.pos.y)) / self.nodes.len() as f32;
    }

    pub fn muscle_extents(&self) -> Array1<f32> {
        return self.muscles.iter()
                .map(|muscle| distance(&self.nodes[muscle.from].pos,
                                       &self.nodes[muscle.to].pos) / muscle.natural_length)
                .collect();
    }

    pub fn apply_anchor_coefficients(&mut self, anchor_coefficients: &Array1<f32>) {
        for (i,coeff) in anchor_coefficients.iter().enumerate() {
            debug_assert!(0.0 <= *coeff && *coeff <= 1.0);
            self.nodes[i].anchor = *coeff;
        }
    }

    pub fn apply_muscle_extents(&mut self, muscle_expansions: &Array1<f32>) {
        for (i,coeff) in muscle_expansions.iter().enumerate() {
            debug_assert!(0.0 <= *coeff && *coeff <= 2.0);
            self.muscles[i].extension_factor = *coeff;
        }
    }
}

pub struct BodyState {
    bodies: Arena<Body>
}

impl BodyState {
    pub fn new() -> BodyState {
        return BodyState {
            bodies: Arena::new()
        };
    }

    pub fn spawn_at(&mut self, spawn: &Point2<f32>, vel: &Vector2<f32>, genome: &Genome) -> Index {
        return self.bodies.insert(Body {
            nodes: genome.nodes.iter().map(|v| Node { pos: spawn + v, vel: *vel, anchor: 0.0 }).collect(),
            muscles: genome.muscles.clone().iter()
                .map(|(a, b)| Muscle {
                    from: *a, to: *b,
                    natural_length: distance(&(spawn + genome.nodes[*a]), &(spawn + genome.nodes[*b])),
                    extension_factor: 1.0
                }).collect(),
            energy: START_ENERGY
        });
    }

    pub fn get(&self, idx: Index) -> Option<&Body> {
        return self.bodies.get(idx);
    }

    pub fn get_mut(&mut self, idx: Index) -> Option<&mut Body> {
        return self.bodies.get_mut(idx);
    }

    pub fn remove(&mut self, idx: Index) -> Option<Body> {
        return self.bodies.remove(idx);
    }
}

pub fn apply_body_forces(bodies: &mut BodyState, dt: f32) {

    for (_id, mut b) in bodies.bodies.iter_mut() {
        for muscle in b.muscles.iter() {
            {
                // Apply Muscle forces.
                let delta = &b.nodes[muscle.from].pos - &b.nodes[muscle.to].pos;
                let current_length = delta.norm();

                let force = if b.energy >= 0.0 {
                    b.energy -= (muscle.natural_length * muscle.extension_factor - muscle.natural_length).abs() * dt * MOVE_ENERGY;
                    muscle.natural_length * muscle.extension_factor - current_length
                } else {
                    (muscle.natural_length - current_length) / 2.0
                };

//                eprintln!("{}", force);

                b.nodes[muscle.from].vel += delta.normalize() * force * dt;
                b.nodes[muscle.to].vel -= delta.normalize() * force * dt;
            }

            {
                // Dampen muscle oscillation
                let v_a: Vector2<f32> = b.nodes[muscle.from].vel;
                let v_b: Vector2<f32> = b.nodes[muscle.to].vel;
                let mid = (v_a + v_b) / 2.0;

                b.nodes[muscle.from].vel += (mid - v_a) * dt * 0.2;
                b.nodes[muscle.to].vel += (mid - v_b) * dt * 0.2;
            }
        }

        for node in b.nodes.iter_mut() {
            let dv = node.vel * node.anchor;
            node.vel *= (1.0 - node.anchor).powf(dt);
//            if node.anchor > 0.5 {
//                node.vel *= 0.0;
//            }
        }

    }
}

pub fn integrate_positions(bodies: &mut BodyState, dt: f32) {
    for (_id, ref mut b) in bodies.bodies.iter_mut() {
        for node in b.nodes.iter_mut() {
            node.pos += node.vel * dt;
        }
    }
}