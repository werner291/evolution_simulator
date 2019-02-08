mod ffnn;
mod genome;
mod sim_framework;
mod food;

use std::time::Duration;
use std::prelude::v1::Vec;
use sdl2::rect::Point;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::render::Canvas;
use sdl2::video::Window;
use sdl2::gfx::primitives::DrawRenderer;
use genome::Genome;
//use std::collections::vec_deque::VecDeque;
use sim_framework::BodyState;
use nalgebra::{Vector2, Point2};
use generational_arena::Index;
use std::collections::vec_deque::VecDeque;
use crate::sim_framework::START_ENERGY;
use rand::Rng;
use ndarray::{Array1, stack, s, Axis, array};
use crate::food::FoodState;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;
const SCALE: f32 = 4.0;
const DT: f32 = 0.1;
const TIME_ENERGY_CONSUMPTION: f32 = 0.2;

const REPRODUCTION_ENERGY: f32 = 90.0;
const FOOD_AMOUNT: usize = 100000 / (SCALE * SCALE) as usize;
const NUM_ORGANISMS: usize = 100;

const FOOD_ISLANDS: usize = 10;
const FOOD_RATE: u32 = 10;
const FOOD_ISLAND_RADIUS: f32 = 5.0;
const FOOD_ISLAND_LIFETIME: u32 = 800;

pub struct Organism {
    genome: Genome,
    food_sensors: Array1<f32>,
    body: Index,
    times_reproduced: usize,
    trail: VecDeque<Point2<f32>>,
}

#[derive(Debug)]
struct Decisions {
    anchor_coefficients: Array1<f32>,
    muscle_expansions: Array1<f32>,
    try_to_reproduce: bool,
}

//impl Organism {
//
//    fn think(&mut self, muscle_tensions: &Array1<f32>) -> Decisions {
//        let brain_in =
//            self.nodes.iter().map(|n| &n.food_sensor)
//                .chain(muscle_tensions.iter())
//                .chain([self.energy / MAX_ENERGY].iter()).cloned().collect();
//
//        let brain_out = self.genome.brain.compute(brain_in);
//
//        let n_nodes = self.genome.nodes.len();
//        let n_conns = self.alive_connections.len();
//
////        eprintln!("{:?}", brain_out);
//
//        return Decisions {
//            anchor_coefficients: brain_out.slice(s![0..n_nodes]).to_owned(),
//            muscle_expansions: brain_out.slice(s![n_nodes..n_nodes+n_conns]).to_owned(),
//            try_to_reproduce: brain_out[n_nodes + n_conns] > 0.0,
//        };
//    }
//}
//
fn try_reproduce(orgs: &mut Vec<Organism>, bodies: &mut BodyState) {
    let mut new_spawns = Vec::new();

    let mut rng = rand::thread_rng();

    for org in orgs.iter_mut() {
        let parent_body = bodies.get_mut(org.body).expect("To have body.");

        if parent_body.energy > REPRODUCTION_ENERGY {

            let mut rng = rand::thread_rng();

            parent_body.energy -= REPRODUCTION_ENERGY;
            org.times_reproduced += 1;

            let mut gen = org.genome.clone();
            gen.mutate();

            new_spawns.push((gen, parent_body.center()));
        }
    }

    for (gen,pos) in new_spawns {
        let angle: f32 = rng.gen_range(0.0, 2.0 * std::f32::consts::PI);

        let body = bodies.spawn_at(&pos, &Vector2::new(rng.gen_range(-0.1, 0.1), rng.gen_range(-0.1, 0.1)), &gen);

        let mut offspring =
            Organism {
                food_sensors: Array1::zeros(gen.nodes.len()),
                genome: gen,
                body,
                times_reproduced: 0,
                trail: VecDeque::new(),
            };

            offspring.trail.push_back(pos);

        orgs.push(offspring);

    }
//    orgs.append(&mut new_spawns);

}

//    fn spawn_food_middleline(&mut self) {
//        let mut rng = rand::thread_rng();
//
//        if self.food.size() < FOOD_AMOUNT {
//            self.food.insert(
//                Point2::new(WIDTH / SCALE * 0.5, rng.gen_range(0.0, HEIGHT / SCALE))
//            );
//        }
//    }
//
//    fn step(&mut self) {
//        const DT: f32 = 0.2;
//
//        self.time += DT;
//        self.step += 1;
//
//        let mut rng = rand::thread_rng();
//
//        let mut new_spawns = Vec::new();
//
//        for mut org in self.organisms.iter_mut() {
//            let c = org.center();
//
//            if self.step % 30 == 0 {
//                org.trail.push_back(c);
//            }
//
//            if org.trail.len() > 50 {
//                org.trail.pop_front();
//            }
//
//            let muscle_tensions = org.alive_connections.iter()
//                .map(|(c_a, c_b, nat_len)| (distance(&org.nodes[*c_a].pos, &org.nodes[*c_b].pos) / nat_len) - 1.0).collect();
//
//            let decisions = org.think(&muscle_tensions);
//
//            apply_muscle_forces(&mut org, &decisions.muscle_expansions);
//
//            // Energy use or just existing / aging.
//            org.energy = (org.energy - TIME_ENERGY_CONSUMPTION * DT * org.genome.maintenance_price() as f32).max(0.0);
//
//            apply_anchorings(org, &decisions.anchor_coefficients);
//
//            for node in org.nodes.iter_mut() {
//                node.pos += node.vel * DT;
//            }
//
//            eat_food(&mut org, &mut self.food);
//
//            if decisions.try_to_reproduce {
//                if let Some(o) = try_reproduce(org) {
//                    new_spawns.push(o) ;
//                }
//            }
//        }
//
//        self.organisms.append(&mut new_spawns);
//
//        self.organisms.retain(|x| x.energy > 0.0);
//
//        if self.organisms.len() == 0 {
//            for _i in 0..100 {
//                self.spawn_seed();
//            }
//        }
//
//        self.food_island_spawns();
//
//        self.apply_threadmill();
//    }
//
//    fn apply_threadmill(&mut self) {
//        let dx = self.time / 10000.0;
//
//        for organism in self.organisms.iter_mut() {
//            for node in organism.nodes.iter_mut() {
//                node.pos.x -= dx;
//            }
//        }
//
//        let mut food_new = RTree::new();
//
//        for food in self.food.iter() {
//            if food.x > 10.0 {
//                food_new.insert(Point2::new(food.x - dx, food.y));
//            }
//        }
//
//        self.food = food_new;
//    }

struct FoodSpawner {
    pos: Point2<f32>,
    start_pos: Point2<f32>,
    seed: u32,
}

fn draw_organisms(organisms: &Vec<Organism>, bodies: &BodyState, canvas: &mut Canvas<Window>)
{
    for org in organisms.iter() {
        let body = &bodies.get(org.body).expect("Organism refers to non-existant body.");

        let rel_energy = (body.energy * 255.0 / START_ENERGY).min(255.0).max(0.0);
        for (node, food) in body.nodes.iter().zip(org.food_sensors.iter()) {
            canvas.filled_circle((node.pos.x * SCALE) as i16,
                                 (node.pos.y * SCALE) as i16,
                                 (node.anchor * 5.0) as i16,
                                 (org.genome.color[0],
                                  org.genome.color[1],
                                  org.genome.color[2], 255u8)).expect("Failed to draw node.");

            if *food > 0.0 {
                canvas.circle((node.pos.x * SCALE) as i16,
                              (node.pos.y * SCALE) as i16,
                              (food.sqrt() * 5.0) as i16,
                              (255, 0, 255, 255u8)).expect("Failed to draw node.");
            }
        }

        canvas.set_draw_color((255, rel_energy as u8, rel_energy as u8));

        for muscle in body.muscles.iter() {
            canvas.draw_line(Point::new((body.nodes[muscle.from].pos.x * SCALE) as i32,
                                        (body.nodes[muscle.from].pos.y * SCALE) as i32),
                             Point::new((body.nodes[muscle.to].pos.x * SCALE) as i32,
                                        (body.nodes[muscle.to].pos.y * SCALE) as i32)).expect("Draw line failed.");
        }

        if org.times_reproduced >= 1 {
            canvas.string((body.nodes[0].pos.x * SCALE) as i16,
                          (body.nodes[0].pos.y * SCALE) as i16,
                          &(org.times_reproduced.to_string() + " " + &((org.genome.mutation_rate * 100.0).round() / 100.0).to_string()),
                          (255, 255, 255, 255)).expect("Can't draw text.")
        }

        canvas.set_draw_color((org.genome.color[0] / 2, org.genome.color[1] / 2, org.genome.color[2] / 2, 128u8));

        canvas.draw_lines(org.trail.iter().map(|p| {
            Point::new((p.x * SCALE) as i32, (p.y * SCALE) as i32)
        }).collect::<Vec<_>>().as_slice()).expect("Cannot draw trail.");
    }
}

fn draw_food(food: &FoodState, canvas: &mut Canvas<Window>) {
    canvas.set_draw_color((0u8,255,0,255));
    canvas.draw_points(food.food.iter().map(|p| Point::new((p.x * SCALE) as i32, (p.y * SCALE) as i32))
        .collect::<Vec<_>>().as_slice()).expect("Failed to draw points.");
}


fn interpret_output(brain_out: &Array1<f32>, genome: &Genome) -> Decisions {
    let n_nodes = genome.nodes.len();
    let n_conns = genome.muscles.len();

    return Decisions {
        anchor_coefficients: brain_out.slice(s![0..n_nodes]).map(|x| x.min(1.0).max(0.0)).to_owned(),
        muscle_expansions: brain_out.slice(s![n_nodes..n_nodes+n_conns]).map(|x| x.max(1.0) + 1.0).to_owned(),
        try_to_reproduce: brain_out[n_nodes + n_conns] > 0.0,
    };
}

fn food_island_spawns(spawners: &mut Vec<FoodSpawner>,
                      food: &mut FoodState,
                      t: f32) {

    let mut rng = rand::thread_rng();

    for mut spawner in spawners.iter_mut() {
        if rng.gen_weighted_bool(FOOD_ISLANDS as u32 / FOOD_RATE) {
            food.food.insert(Point2::new(spawner.pos.x + rng.gen_range(-FOOD_ISLAND_RADIUS, FOOD_ISLAND_RADIUS),
                                         spawner.pos.y + rng.gen_range(-FOOD_ISLAND_RADIUS, FOOD_ISLAND_RADIUS)));
        }

        if rng.gen_weighted_bool(FOOD_ISLAND_LIFETIME) {
            spawner.pos.x = rng.gen_range(0.0, WIDTH / SCALE);
            spawner.pos.y = rng.gen_range(0.0, HEIGHT / SCALE);
        }

        let ang = ((spawner.seed % 47) as f32 + if spawner.seed % 2 == 0 { t } else { -t }) * 0.01 * (spawner.seed % 10) as f32;
        spawner.pos.x = spawner.start_pos.x + 50.0 * ang.cos();
        spawner.pos.y = spawner.start_pos.y + 50.0 * ang.sin();
    }
}

fn init_organisms(body_service: &mut BodyState) -> Vec<Organism> {
    let mut organisms = Vec::new();

    let mut rng = rand::thread_rng();

    for _i in 0..NUM_ORGANISMS {
        let mut genome = Genome::initial();
        genome.mutate();

        let body = body_service.spawn_at(&Point2::new(rng.gen_range(0.0, WIDTH/SCALE), rng.gen_range(0.0, HEIGHT/SCALE)),
                                         &Vector2::new(rng.gen_range(-0.1, 0.1), rng.gen_range(-0.1, 0.1)), &genome);

        organisms.push(
            Organism {
                food_sensors: Array1::zeros(genome.nodes.len()),
                genome,
                body,
                times_reproduced: 0,
                trail: VecDeque::new(),
            }
        );
    }

    return organisms;
}

fn update_organisms(organisms: &mut Vec<Organism>, bodies: &mut BodyState) {
    organisms.retain(|org| {
        let ref mut org_body = bodies.get_mut(org.body).expect("Body exists.");

        let food_sensors = &org.food_sensors;
        let muscle_states = org_body.muscle_extents() - 1.0;

        let brain_input = stack![Axis(0), food_sensors.view(), muscle_states.view(), array![org_body.energy.tanh()]];

        let brain_output = interpret_output(&org.genome.brain.compute(brain_input), &org.genome);

        org_body.apply_anchor_coefficients(&brain_output.anchor_coefficients);
        org_body.apply_muscle_extents(&brain_output.muscle_expansions);

        if org_body.energy <= 0.0 {
            bodies.remove(org.body);
            return false;
        } else {
            return true;
        }
    });
}

fn init_food_spawners() -> Vec<FoodSpawner> {
    let mut rng = rand::thread_rng();

    return (0..FOOD_ISLANDS)
        .map(|_| {
            let pos = Point2::new(rng.gen_range(0.25 * WIDTH / SCALE, 0.75 * WIDTH / SCALE),
                                  rng.gen_range(0.25 * HEIGHT / SCALE, 0.75 * HEIGHT / SCALE));
            return FoodSpawner {
                pos: pos.clone(),
                start_pos: pos,
                seed: rng.gen()
            };
        }).collect();
}

pub fn run(update: &mut FnMut(&mut Canvas<Window>) -> ()) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("rust-sdl2 demo", 800, 600)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();

    'running: loop {

        // Drawing
        canvas.set_draw_color((0, 0, 0, 255u8));
        canvas.clear();

        update(&mut canvas);

        canvas.present();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running;
                }
                _ => {}
            }
        }

        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}

fn organism_life_consume(organisms: &mut Vec<Organism>, bodies: &mut BodyState, dt: f32) {
    for org in organisms.iter_mut() {
        let mut body = bodies.get_mut(org.body).unwrap();

        body.energy = (body.energy - TIME_ENERGY_CONSUMPTION * dt * org.genome.maintenance_price()).max(0.0);
    }
}

pub fn main() {
    let mut bodies = BodyState::new();
    let mut food = FoodState::new();
    let mut organisms = init_organisms(&mut bodies);
    let mut food_spawners = init_food_spawners();
    let mut t = 0.0;

    run(&mut |mut canvas| {
        t += DT;
        if t > 100.0 {
            sim_framework::apply_body_forces(&mut bodies, DT);
            sim_framework::integrate_positions(&mut bodies, DT);
            update_organisms(&mut organisms, &mut bodies);
            organism_life_consume(&mut organisms, &mut bodies, DT);

            try_reproduce(&mut organisms, &mut bodies);

            food.eat_food(&mut organisms, &mut bodies);
            food_island_spawns(&mut food_spawners, &mut food, t);
            draw_food(&food, &mut canvas);

            for spawner in food_spawners.iter() {
                canvas.filled_circle((spawner.pos.x * SCALE) as i16,
                                     (spawner.pos.y * SCALE) as i16,
                                     5,
                                     (0, 255, 0, 128)).expect("Het is allemaal kut.");
            }

            draw_organisms(&organisms, &bodies, &mut canvas);
        }
    });
}