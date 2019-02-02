extern crate sdl2;
extern crate chrono;
extern crate nalgebra as na;
extern crate rand;
extern crate ndarray as nd;
extern crate ndarray;
extern crate spade;
extern crate generational_arena;

mod ffnn;

use std::time::Duration;
use na::{Point2, Vector2, distance};
use std::prelude::v1::Vec;
use rand::Rng;
use nd::{Array1, s};
use sdl2::rect::Point;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::render::Canvas;
use sdl2::video::Window;
use sdl2::gfx::primitives::DrawRenderer;
use ffnn::FFNN;
use spade::rtree::RTree;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;
const SCALE: f32 = 2.0;
const DT: f32 = 0.01;
const MUTATION_RATE: f32 = 0.2;
const START_ENERGY: f32 = 10.0;
const MAX_ENERGY: f32 = 100.0;
const EAT_RADIUS: f32 = 1.0;
const FOOD_ENERGY: f32 = 10.0;
const REPRODUCTION_ENERGY: f32 = 50.0;
const FOOD_AMOUNT: usize = 2000;
const TIME_ENERGY_CONSUMPTION : f32 = 0.1;

#[derive(Debug)]
struct Genome {
    nodes: Vec<Vector2<f32>>,
    connections: Vec<(usize, usize)>,
    brain: FFNN,
    color: [u8; 3],
    mutation_rate: f32
}

impl Genome {
    pub fn initial() -> Genome {
        let nodes = vec![Vector2::new(-1.0, 1.0),
                         Vector2::new(0.0, 0.0),
                         Vector2::new(1.0, 0.0)];

        let connections = vec![(0, 1), (1, 2), (2, 0)];

        let brain_input = Genome::brain_input_size(nodes.len(), connections.len());
        let brain_output = Genome::brain_output_size(nodes.len(), connections.len()); // Just the initial muscle thingy

        let mut rng = rand::thread_rng();

        return Genome {
            nodes,
            connections,
            brain: FFNN::initial(brain_input, brain_output),
            color: [rng.gen_range(0,255),rng.gen_range(0,255),rng.gen_range(0,255)],
            mutation_rate: rng.gen_range(0.1,10.0)
        };
    }

    fn brain_input_size(anchors: usize, muscles: usize) -> usize {
        return anchors // Food sensors
            + muscles // Muscle tension
            + 1 // Energy level
            + 1; // Bias;
    }

    fn brain_output_size(anchors: usize, muscles: usize) -> usize {
        return anchors
            + muscles
            + 1; // Reproduction output )
    }

    pub fn mutated_copy(&self) -> Genome {
        let mut rng = rand::thread_rng();
        const base_mut : f32 = 1.0;
        let mutation_coeff : f32 = base_mut.powf(self.mutation_rate);

        let mut nodes: Vec<Vector2<f32>> = self.nodes.iter()
            .map(|x| Vector2::new(x.x + rng.gen_range(-mutation_coeff, mutation_coeff),
                                  x.y + rng.gen_range(-mutation_coeff, mutation_coeff))).collect();

        let mut connections = self.connections.clone();

        if rng.gen_weighted_bool(10) {
            let attach_to = rng.gen_range(0, nodes.len());

            let node_pos = nodes[attach_to] + Vector2::new(rng.gen_range(-mutation_coeff, mutation_coeff),
                                                           rng.gen_range(-mutation_coeff, mutation_coeff));

            nodes.push(node_pos);

            connections.push((nodes.len() - 1, attach_to));
        }

        return Genome {
            brain: self.brain.mutated_copy(MUTATION_RATE,
                                           Genome::brain_input_size(nodes.len(), connections.len()),
                                           Genome::brain_output_size(nodes.len(), connections.len())),
            connections,
            nodes,
            color: [self.color[0] + rng.gen_range(0,10),
                    self.color[1] + rng.gen_range(0,10),
                    self.color[2] + rng.gen_range(0,10)],
            mutation_rate: self.mutation_rate + rng.gen_range(-0.1,0.1)
        };

    }
}

struct Node {
    pos: Point2<f32>,
    vel: Vector2<f32>,
    anchor: f32,
    food_sensor: f32,
}

struct Strut {
    node_a: usize,
    node_b: usize,
    relaxation_length: f32,
    target_length: f32,
}

struct Organism {
    genome: Genome,
    nodes: Vec<Node>,
    alive_connections: Vec<(usize, usize, f32)>,
    energy: f32,
    times_reproduced: usize
}

struct Decisions {
    anchor_coefficients: Array1<f32>,
    muscle_expansions: Array1<f32>,
    try_to_reproduce: bool,
}

impl Organism {
    fn spawn_at(spawn: &Point2<f32>, vel: &Vector2<f32>, genome: Genome) -> Organism {
        return Organism {
            nodes: genome.nodes.iter().map(|v| Node { food_sensor: 0.0, pos: spawn + v, vel: *vel, anchor: 0.0 }).collect(),
            alive_connections: genome.connections.clone().iter()
                .map(|(a, b)| (*a, *b, distance(&(spawn + genome.nodes[*a]), &(spawn + genome.nodes[*b]))))
                .collect(),
            energy: START_ENERGY,
            genome,
            times_reproduced: 0
        };
    }

    fn center(&self) -> Point2<f32> {
        return self.nodes.iter().fold(Point2::new(0.0,0.0), |p1,p2| Point2::new(p1.x+p2.pos.x,p1.y+p2.pos.y)) / self.nodes.len() as f32;
    }

    fn think(&mut self, muscle_tensions: &Array1<f32>) -> Decisions {
        let brain_in =
            self.nodes.iter().map(|n| &n.food_sensor)
                .chain(muscle_tensions.iter())
                .chain([self.energy / MAX_ENERGY, 1.0].iter()).cloned().collect();

        let brain_out = self.genome.brain.compute(brain_in);

//        debug_assert_eq!(brain_out.len(), self.genome.nodes.len() + self.genome.connections.len());

        let n_nodes = self.genome.nodes.len();
        let n_conns = self.alive_connections.len();

        return Decisions {
            anchor_coefficients: brain_out.slice(s![0..n_nodes]).to_owned(),
            muscle_expansions: brain_out.slice(s![n_nodes..n_nodes+n_conns]).to_owned(),
            try_to_reproduce: brain_out[n_nodes + n_conns] > 0.0,
        }
    }
}

impl State {
    fn new() -> State {
        let mut organisms = Vec::new();

        let mut food = RTree::new();

        let mut s = State {
            organisms,
            food,
            time:0.0
        };

        s.initialize();

        return s;
    }

    fn initialize(&mut self) {
        const NUM_ORGANISMS: usize = 100;

        for _i in 0..NUM_ORGANISMS {
            self.spawn_seed();
        }

        let mut rng = rand::thread_rng();
        for _i in 0..FOOD_AMOUNT {
            self.food.insert(Point2::new(rng.gen_range(0.0, WIDTH / SCALE),
                                         rng.gen_range(0.0, HEIGHT / SCALE)));
        }
    }

    fn spawn_seed(&mut self) {
        let mut rng = rand::thread_rng();
        self.organisms.push(
            Organism::spawn_at(&Point2::new(
                rng.gen_range(0.0, WIDTH / SCALE),
                rng.gen_range(0.0, HEIGHT / SCALE),
            ),
                               &Vector2::new(
                                   rng.gen_range(-0.1, 0.1),
                                   rng.gen_range(-0.1, 0.1),
                               )
                               , Genome::initial()));
    }

    fn spawn_food(&mut self) {

        let mut rng = rand::thread_rng();

        let rot = self.time / 100.0;

        if self.food.size() < FOOD_AMOUNT {

            self.food.insert(
                Point2::new(
                    (WIDTH/2.0) / SCALE + rot.sin() * 20.0 + rng.gen_range(-15.0,15.0),
                    (HEIGHT/2.0) / SCALE + rot.cos() * 20.0 + rng.gen_range(-15.0,15.0)
                )
            );
        }
    }

    fn step(&mut self, delta: f32) {
        const DT: f32 = 0.2;

        self.time += DT;

        let mut rng = rand::thread_rng();

        let mut new_spawns = Vec::new();

        for mut org in self.organisms.iter_mut() {
            let muscle_tensions = org.alive_connections.iter()
                .map(|(c_a, c_b, nat_len)| (distance(&org.nodes[*c_a].pos, &org.nodes[*c_b].pos) / nat_len) - 1.0).collect();

            let decisions = org.think(&muscle_tensions);

            for i in 0..org.alive_connections.len() {
                let (c_a, c_b, natural_length) = org.alive_connections[i];

                let delta = &org.nodes[c_a].pos - &org.nodes[c_b].pos;
                let current_length = delta.norm();

                let force = if org.energy >= 0.0 {
                    natural_length * (1.0 + decisions.muscle_expansions[i].tanh() / 4.0) - current_length
                } else {
                    (natural_length - current_length) / 2.0
                };

                org.energy = (org.energy - force.abs() * DT * 0.01).max(0.0);

                // TODO: Too much force breaks the muscle

                org.nodes[c_a].vel += delta.normalize() * force * DT;
                org.nodes[c_b].vel -= delta.normalize() * force * DT;

                let mid = (org.nodes[c_a].vel + org.nodes[c_b].vel) / 2.0;
                org.nodes[c_a].vel += (mid - org.nodes[c_a].vel) * DT * 0.1;
                org.nodes[c_b].vel += (mid - org.nodes[c_b].vel) * DT * 0.1;
            }

            org.energy = (org.energy - TIME_ENERGY_CONSUMPTION * DT * org.nodes.len() as f32).max(0.0);

            for i in 0..org.nodes.len() {
                org.nodes[i].anchor = decisions.anchor_coefficients[i].max(0.0).min(1.0);
                org.nodes[i].vel -= org.nodes[i].vel * decisions.anchor_coefficients[i].max(0.0).min(1.0) * DT;
            }

            for node in org.nodes.iter_mut() {
                node.pos += node.vel * DT;
            }

            for node in org.nodes.iter_mut() {

                node.food_sensor *= 0.9;

                let eaten: Vec<Point2<f32>> = self.food.lookup_in_circle(&node.pos, &EAT_RADIUS).iter().map(|x| Point2::new(x.x, x.y)).collect();
                for i in 0..eaten.len() {
                    self.food.remove(&eaten[i]);
                    org.energy = (org.energy + FOOD_ENERGY).min(MAX_ENERGY);
                    node.food_sensor += 1.0;
                }
            }

            if decisions.try_to_reproduce && org.energy > REPRODUCTION_ENERGY {
                org.energy -= REPRODUCTION_ENERGY;
                org.times_reproduced += 1;

                let angle : f32 = rng.gen_range(0.0, 2.0 * std::f32::consts::PI);

                new_spawns.push(
                    Organism::spawn_at(&(org.center() + Vector2::new(angle.cos(),angle.sin()) * 10.0), &Vector2::new(
                        rng.gen_range(-0.1, 0.1),
                        rng.gen_range(-0.1, 0.1),
                    ), org.genome.mutated_copy())
                );
            }
        }

        self.organisms.append(&mut new_spawns);

        self.organisms.retain(|x| x.energy > 0.0);

        if self.organisms.len() <= 5 {
            self.spawn_seed();
        }

        self.spawn_food();
    }

    fn draw(&self, canvas: &mut Canvas<Window>) {
        for org in self.organisms.iter() {
            let rel_energy = (org.energy * 255.0 / START_ENERGY).min(255.0);

            for node in org.nodes.iter() {
                canvas.filled_circle((node.pos.x * SCALE) as i16,
                                     (node.pos.y * SCALE) as i16,
                                     (node.anchor * 5.0) as i16,
                                     (org.genome.color[0],org.genome.color[1],org.genome.color[2], 255u8)).expect("Failed to draw node.");

                if (node.food_sensor > 0.0) {
                    canvas.circle((node.pos.x * SCALE) as i16,
                                  (node.pos.y * SCALE) as i16,
                                  (node.food_sensor.sqrt() * 5.0) as i16,
                                  (255, 0, 255, 255u8)).expect("Failed to draw node.");
                }
            }

            canvas.set_draw_color((255, rel_energy as u8, rel_energy as u8));

            for (i, j, _) in org.alive_connections.iter() {
                canvas.draw_line(Point::new((org.nodes[*i].pos.x * SCALE) as i32, (org.nodes[*i].pos.y * SCALE) as i32),
                                 Point::new((org.nodes[*j].pos.x * SCALE) as i32, (org.nodes[*j].pos.y * SCALE) as i32)).expect("Draw line failed.");
            }

            if org.times_reproduced >= 1 {
                canvas.string((org.nodes[0].pos.x * SCALE) as i16,
                              (org.nodes[0].pos.y * SCALE) as i16,
                              &(org.times_reproduced.to_string() + " " + &((org.genome.mutation_rate * 100.0).round()/100.0).to_string()),
                              (255, 255, 255, 255)).expect("Can't draw text.")
            }
        }

        canvas.set_draw_color((0, 255, 0));

        let rot = self.time / 100.0;

        canvas.draw_line(Point::new((WIDTH/2.0) as i32,
                                    (HEIGHT/2.0) as i32),
                         Point::new((WIDTH/2.0 + rot.sin() * 30.0 * SCALE) as i32,
                                    (HEIGHT/2.0 + rot.cos() * 30.0 * SCALE) as i32)).expect("Beep boop.");

        canvas.draw_points(self.food.iter().map(|p| Point::new((p.x * SCALE) as i32, (p.y * SCALE) as i32))
            .collect::<Vec<_>>().as_slice()).expect("Failed to draw points.");
    }
}

struct State {
    organisms: Vec<Organism>,
    food: RTree<Point2<f32>>,

    time: f32,
}

pub fn main() {
    let sdl_context = sdl2::init().unwrap();

    let video_subsystem = sdl_context.video().unwrap();

    let mut rng = rand::thread_rng();

    let mut organisms: State = State::new();

    let window = video_subsystem.window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();

    let mut age = 0;
    let mut generation_time = 0;

    'running: loop {
        canvas.set_draw_color((0, 0, 0));
        canvas.clear();

        organisms.step(DT);

        organisms.draw(&mut canvas);

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running;
                }
                _ => {}
            }
        }

        canvas.present();
    }
}