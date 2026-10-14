use clap::Parser;
use std::error::Error;
use std::path::PathBuf;
use persistence_images::{
    dataset::{SpacingDiagram,load_dataset,to_scaled_diagrams},
    image::{compute_bounds,Bounds}, io,
    stable::stable_batch_images,
    optimized::{Engine,batch_images},
};

#[derive(Debug,Clone,Copy,PartialEq,Eq,clap::ValueEnum)]
enum Representation { Sampled, Stable }

#[derive(Debug,Parser)]
#[command(author,version,about="Persistence images from CSV persistence diagrams")]
struct Args {
    #[arg(short,long)] input:PathBuf,
    #[arg(short,long)] spacing:PathBuf,
    #[arg(short,long)] output:PathBuf,
    #[arg(long,default_value_t=10)] nx:usize,
    #[arg(long,default_value_t=8)] ny:usize,
    #[arg(long,default_value_t=0.86)] sigma_x:f64,
    #[arg(long,default_value_t=0.63984)] sigma_y:f64,
    #[arg(long,default_value_t=false)] remove_h0:bool,
    /// sampled: legacy unweighted pixel centers; stable: weighted Gaussian pixel integrals
    #[arg(long,value_enum,default_value_t=Representation::Sampled)]
    representation:Representation,
    /// Saturation scale for the stable persistence ramp weight.
    #[arg(long,default_value_t=1.0)] weight_scale:f64,
    /// Fixed image grid boundaries: xmin xmax ymin ymax.
    #[arg(long,num_args=4)] bounds:Option<Vec<f64>>,
    #[arg(long)] diagram_suffix:Option<String>,
    /// auto, direct, separable, shared, sparse, or interp-fft
    #[arg(long,value_enum,default_value_t=Engine::Auto)] engine:Engine,
    /// Optional absolute approximation error per pixel; required for sparse and interp-fft.
    /// Auto is exact unless this argument is supplied.
    #[arg(long)] absolute_error:Option<f64>,
}
fn main()->Result<(),Box<dyn Error+Send+Sync>>{
    let args=Args::parse();
    let dataset:Vec<SpacingDiagram>=load_dataset(&args.input,&args.spacing,
        Some(&args.output),args.diagram_suffix.as_deref(),args.remove_h0)?;
    if dataset.is_empty(){ println!("No data found."); return Ok(()); }
    let diagrams=to_scaled_diagrams(&dataset);
    let fixed_bounds=args.bounds.as_ref().map(|v|Bounds{
        x_min:v[0],x_max:v[1],y_min:v[2],y_max:v[3]
    });
    let bounds=match fixed_bounds.or_else(||compute_bounds(&diagrams)){
        Some(b)=>b,
        None=>{eprintln!("No persistence points remain after preprocessing; provide --bounds for empty stable images.");return Ok(());}
    };
    println!("Bounds: {:?}",bounds);
    let spacings:Vec<f64>=dataset.iter().map(|d|d.voxel_spacing).collect();
    let images=if args.representation==Representation::Stable {
        if args.absolute_error.is_some() {
            return Err("Stable representation does not accept --absolute-error".into());
        }
        if !matches!(args.engine,Engine::Auto|Engine::Direct|Engine::Separable) {
            return Err("Stable representation supports --engine auto, direct, or separable".into());
        }
        stable_batch_images(&diagrams,&spacings,bounds,args.nx,args.ny,
            args.sigma_x,args.sigma_y,args.weight_scale)
    } else {
        batch_images(&diagrams,&spacings,bounds,args.nx,args.ny,
            args.sigma_x,args.sigma_y,args.engine,args.absolute_error)
    }.map_err(|e|->Box<dyn Error+Send+Sync>{e.into()})?;
    let filenames:Vec<String>=dataset.iter().map(|d|io::filename_only(&d.path)).collect();
    io::save_images_to_csv(&filenames,&images,&args.output)?;
    println!("Saved to {:?}",args.output);
    Ok(())
}