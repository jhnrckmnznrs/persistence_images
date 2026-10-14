//! Benchmark stable direct CDF integration versus cross-diagram shared bases.
//! cargo run --release --locked --example benchmark_stable_shared
use persistence_images::{Bounds,Engine,stable_persistence_image,stable_batch_images_with_engine};
use std::time::Instant;
use std::hint::black_box;
fn main(){
    let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
    let diagrams:Vec<Vec<(f64,f64)>>=(0..12).map(|frame|
        (0..1500).map(|k|(
            ((k*7+frame)%24) as f64/24.0,
            ((k*11+frame)%16+1) as f64/17.0
        )).collect()
    ).collect();
    let spacings=vec![1.0;diagrams.len()];
    let start=Instant::now();
    let reference:Vec<_>=diagrams.iter().map(|d|black_box(
        stable_persistence_image(d,b,80,64,0.14,0.11,0.7).unwrap()
    )).collect();
    let direct_time=start.elapsed();
    let start=Instant::now();
    let shared=black_box(stable_batch_images_with_engine(
        &diagrams,&spacings,b,80,64,0.14,0.11,0.7,Engine::Shared).unwrap());
    let shared_time=start.elapsed();
    let max_error=reference.iter().zip(&shared).flat_map(|(a,b)|a.iter().zip(b))
        .map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
    println!("direct={direct_time:?} shared={shared_time:?} max_abs_error={max_error:.3e}");
}
