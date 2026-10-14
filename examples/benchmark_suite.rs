//! Shared deterministic persistence-image benchmark; one (case, engine) per process.
//! cargo run --release --locked --example benchmark_suite -- --case small_continuous --engine separable
use persistence_images::{Bounds,Diagram,Engine,batch_images};
use serde::Deserialize;
use serde_json::json;
use std::{env,fs,time::Instant,collections::BTreeMap};

const WORKLOADS:&str=include_str!("../benchmarks/workloads.json");
const FNV_OFFSET:u64=14695981039346656037;
const FNV_PRIME:u64=1099511628211;

#[derive(Clone,Deserialize)]
struct Case {
 name:String,profile:String,kind:String,n:usize,nx:usize,ny:usize,
 sigma_x:f64,sigma_y:f64,seed:u32,fingerprint_fnv64:String,
}
#[derive(Deserialize)]
struct Suite {
 schema_version:u32,generator:String,bounds:[f64;4],
 quantized_levels:[u32;2],cases:Vec<Case>,
}

fn generate(c:&Case,levels:[u32;2])->(Diagram,String) {
 let mut state=c.seed;
 let mut fingerprint=FNV_OFFSET;
 let mut points=Vec::with_capacity(c.n);
 for _ in 0..c.n {
  let mut pair=[0.0_f64;2];
  for axis in 0..2 {
   state=state.wrapping_mul(1664525).wrapping_add(1013904223);
   let u24=state>>8;
   for shift in [0,8,16,24] {
    fingerprint^=((u24>>shift)&255) as u64;
    fingerprint=fingerprint.wrapping_mul(FNV_PRIME);
   }
   pair[axis]=if c.kind=="continuous" {
    (u24 as f64+0.5)/(1u64<<24) as f64
   } else {
    (u24%levels[axis]) as f64/levels[axis] as f64
   };
  }
  points.push((pair[0],pair[1]));
 }
 (points,format!("{fingerprint:016x}"))
}
fn sample_indices(nx:usize,ny:usize)->Vec<usize> {
 let m=nx*ny;
 if m<=2048 {(0..m).collect()} else {(0..64).map(|i|i*(m-1)/63).collect()}
}
fn reference(pts:&Diagram,c:&Case,indices:&[usize])->Vec<f64> {
 indices.iter().map(|&idx|{
  let x=((idx/c.ny) as f64+0.5)/c.nx as f64;
  let y=((idx%c.ny) as f64+0.5)/c.ny as f64;
  let (mut sum,mut comp)=(0.0_f64,0.0_f64);
  for &(b,p) in pts {
   let (dx,dy)=((x-b)/c.sigma_x,(y-p)/c.sigma_y);
   let v=(-0.5*(dx*dx+dy*dy)).exp();
   let temp=v-comp; let next=sum+temp;
   comp=(next-sum)-temp; sum=next;
  }
  sum
 }).collect()
}
fn peak_bytes()->Option<u64> {
 fs::read_to_string("/proc/self/status").ok().and_then(|s|
 s.lines().find(|l|l.starts_with("VmHWM:"))
 .and_then(|l|l.split_whitespace().nth(1))
 .and_then(|v|v.parse::<u64>().ok()).map(|v|v*1024))
}
fn cpu_model()->Option<String> {
 fs::read_to_string("/proc/cpuinfo").ok().and_then(|s|
 s.lines().find(|l|l.starts_with("model name"))
 .and_then(|l|l.split_once(':').map(|(_,v)|v.trim().to_string())))
}
fn engine(s:&str)->Result<Engine,String> {
 match s {
  "auto"=>Ok(Engine::Auto),"direct"=>Ok(Engine::Direct),
  "separable"=>Ok(Engine::Separable),"shared"=>Ok(Engine::Shared),
  "sparse"=>Ok(Engine::Sparse),"interp-fft"=>Ok(Engine::InterpFft),
  _=>Err(format!("Unknown engine: {s}"))
 }
}
fn run()->Result<(),Box<dyn std::error::Error>> {
 let suite:Suite=serde_json::from_str(WORKLOADS)?;
 if suite.schema_version!=1 {return Err("Invalid benchmark schema version".into());}
 let (mut case,mut method)=("small_continuous".to_string(),"separable".to_string());
 let (mut repeats,mut warmup)=(5usize,1usize);
 let mut epsilon=1e-8_f64;
 let mut output:Option<String>=None;
 let mut args=env::args().skip(1);
 while let Some(flag)=args.next() {
  if flag=="--list" {
   for c in &suite.cases {
    println!("{} [{}] points={} grid={}x{}",c.name,c.profile,c.n,c.nx,c.ny);
   }
   return Ok(());
  }
  let value=args.next().ok_or("Missing CLI argument value")?;
  match flag.as_str() {
   "--case"=>case=value,
   "--engine"=>method=value,
   "--repeats"=>repeats=value.parse()?,
   "--warmup"=>warmup=value.parse()?,
   "--absolute-error"=>epsilon=value.parse()?,
   "--output"=>output=Some(value),
   _=>return Err(format!("Invalid CLI argument: {flag}").into())
  }
 }
 if repeats==0 || !epsilon.is_finite() || epsilon<=0.0 {
  return Err("Invalid repeats or absolute-error".into());
 }
 let c=suite.cases.iter().find(|c|c.name==case).ok_or("Unknown case")?;
 let (points,fingerprint)=generate(c,suite.quantized_levels);
 if fingerprint!=c.fingerprint_fnv64 {
  return Err(format!("Generator fingerprint mismatch {fingerprint}").into());
 }
 let engine=engine(&method)?;
 let bounds=Bounds{x_min:suite.bounds[0],x_max:suite.bounds[1],
                   y_min:suite.bounds[2],y_max:suite.bounds[3]};
 let tol=match engine {
  Engine::Sparse|Engine::InterpFft|Engine::Auto=>Some(epsilon),_=>None
 };
 let compute=||->Result<Vec<f64>,String>{
  let mut images=batch_images(std::slice::from_ref(&points),&[1.0],
    bounds,c.nx,c.ny,c.sigma_x,c.sigma_y,engine,tol)?;
  Ok(images.remove(0))
 };
 for _ in 0..warmup {std::hint::black_box(compute()?);}
 let mut times:Vec<f64>=Vec::with_capacity(repeats);
 let mut image=Vec::new();
 for _ in 0..repeats {
  let start=Instant::now();
  image=std::hint::black_box(compute()?);
  times.push(start.elapsed().as_secs_f64()*1000.0);
 }
 let idx=sample_indices(c.nx,c.ny);
 let truth=reference(&points,c,&idx);
 let err=idx.iter().zip(truth.iter()).map(|(&i,&r)|(image[i]-r).abs())
    .fold(0.0_f64,f64::max);
 let refmax=truth.iter().map(|v|v.abs()).fold(0.0_f64,f64::max);
 let mut sorted=times.clone();
 sorted.sort_by(f64::total_cmp);
 let mid=repeats/2;
 let median=if repeats%2==1 {sorted[mid]}
            else {(sorted[mid-1]+sorted[mid])/2.0};
 let mut threads=BTreeMap::new();
 for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"] {
  threads.insert(key,env::var(key).ok());
 }
 let report=json!({
  "schema_version":1,"suite":"persistence-images-cross-language-v1",
  "case":c.name,"profile":c.profile,"kind":c.kind,"language":"rust",
  "engine":method,"generator":suite.generator,
  "source_fingerprint_fnv64":fingerprint,
  "n_points":c.n,"nx":c.nx,"ny":c.ny,
  "sigma_x":c.sigma_x,"sigma_y":c.sigma_y,
  "absolute_error_requested":tol,"warmup":warmup,"repeats":repeats,
  "times_ms":times,"median_ms":median,
  "peak_rss_bytes":peak_bytes(),
  "memory_metric":"Linux VmHWM process peak (includes runtime and warmup)",
  "reference_scope":if idx.len()==c.nx*c.ny {"full"} else {"64 sampled pixels"},
  "reference_pixel_count":idx.len(),
  "max_abs_error":err,"max_scaled_error":err/refmax.max(1.0),
  "sampled_pixel_indices":idx,
  "sampled_pixel_values":idx.iter().map(|&i|image[i]).collect::<Vec<f64>>(),
  "image_sum":image.iter().sum::<f64>(),
  "image_linf":image.iter().map(|v|v.abs()).fold(0.0_f64,f64::max),
  "rust_package_version":env!("CARGO_PKG_VERSION"),
  "platform":env::consts::OS,"arch":env::consts::ARCH,
  "cpu_model":cpu_model(),
  "logical_cpus":std::thread::available_parallelism().ok().map(|v|v.get()),
  "thread_env":threads,
  "precision_note":"Sampled direct-reference error is not an image-wide bound; FFT roundoff is excluded"
 });
 let text=serde_json::to_string_pretty(&report)?;
 if let Some(file)=output {fs::write(file,format!("{text}\n"))?;}
 println!("{text}");
 Ok(())
}
fn main() {
 if let Err(e)=run(){eprintln!("Benchmark error: {e}");std::process::exit(2);}
}
#[cfg(test)]
mod tests {
 use super::*;
 #[test]
 fn fingerprints() {
  let s:Suite=serde_json::from_str(WORKLOADS).unwrap();
  for c in &s.cases {
   let (d,h)=generate(c,s.quantized_levels);
   assert_eq!(d.len(),c.n);
   assert_eq!(h,c.fingerprint_fnv64);
  }
 }
 #[test]
 fn sample_indexing() {
  assert_eq!(sample_indices(32,24).len(),768);
  assert_eq!(sample_indices(96,80).len(),64);
 }
}
