//! Compare a cached one-FFT approximation with the reference formula.
//! cargo run --release --example benchmark_interp_fft
use persistence_images::{Bounds,Engine,batch_images,persistence_image};
use std::time::Instant;
use std::hint::black_box;
fn main() {
    let b=Bounds{x_min:0.0,x_max:1.0,y_min:0.0,y_max:1.0};
    let d:Vec<(f64,f64)>=(0..20_000).map(|k|
        (((k*48271 % 2147483647) as f64)/2147483647.0,
         ((k*69621 % 2147483647) as f64)/2147483647.0)).collect();
    let t=Instant::now();
    let exact=black_box(persistence_image(&d,b,128,128,0.2,0.18));
    let time_exact=t.elapsed();
    let t=Instant::now();
    let im=black_box(batch_images(&[d],&[1.0],b,128,128,0.2,0.18,
                         Engine::InterpFft,Some(1e-8)).unwrap());
    let time_fft=t.elapsed();
    let err=exact.iter().zip(&im[0]).map(|(a,b)|(a-b).abs())
        .fold(0.0_f64,f64::max);
    println!("reference={time_exact:?}, fft={time_fft:?}, max_abs_difference={err:.3e}");
}
