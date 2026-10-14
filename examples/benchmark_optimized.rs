//! cargo run --release --example benchmark_optimized
//! Local benchmark; results depend on hardware and compiler settings.
use persistence_images::{Bounds,Engine,batch_images,persistence_image};
use std::hint::black_box;
use std::time::Instant;
fn main(){
    let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
    let diagrams:Vec<_>=(0..30).map(|frame|(0..1000).map(|k|
        (((k*7+frame)%24) as f64/24.0,((k*11+frame)%16) as f64/16.0))
        .collect()).collect();
    let spacings=vec![1.0;diagrams.len()];
    let start=Instant::now();
    let reference:Vec<_>=diagrams.iter().map(|d|black_box(
        persistence_image(d,b,64,64,0.07,0.08))).collect();
    let baseline=start.elapsed();
    let start=Instant::now();
    let optimized=black_box(batch_images(&diagrams,&spacings,b,64,64,
        0.07,0.08,Engine::Auto,None).unwrap());
    let elapsed=start.elapsed();
    let error=reference.iter().zip(optimized.iter())
        .flat_map(|(a,b)|a.iter().zip(b.iter()))
        .map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
    println!("direct={baseline:?}; auto={elapsed:?}; max_error={error:.3e}");
}