use clap::Parser;
use std::error::Error;
use std::path::PathBuf;
use persistence_images::{
    dataset::{SpacingDiagram,load_dataset,to_scaled_diagrams},
    image::compute_bounds, io,
    optimized::{Engine,batch_images},
};

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
    #[arg(long)] diagram_suffix:Option<String>,
    /// auto, direct, separable, shared, or sparse
    #[arg(long,value_enum,default_value_t=Engine::Auto)] engine:Engine,
    /// Optional absolute truncation error per pixel; required for sparse.
    /// Auto is exact unless this argument is supplied.
    #[arg(long)] absolute_error:Option<f64>,
}
fn main()->Result<(),Box<dyn Error+Send+Sync>>{
    let args=Args::parse();
    let dataset:Vec<SpacingDiagram>=load_dataset(&args.input,&args.spacing,
        Some(&args.output),args.diagram_suffix.as_deref(),args.remove_h0)?;
    if dataset.is_empty(){ println!("No data found."); return Ok(()); }
    let diagrams=to_scaled_diagrams(&dataset);
    let Some(bounds)=compute_bounds(&diagrams) else{
        eprintln!("No persistence points remain after preprocessing.");return Ok(());
    };
    println!("Bounds: {:?}",bounds);
    let spacings:Vec<f64>=dataset.iter().map(|d|d.voxel_spacing).collect();
    let images=batch_images(&diagrams,&spacings,bounds,args.nx,args.ny,
        args.sigma_x,args.sigma_y,args.engine,args.absolute_error)
        .map_err(|e|->Box<dyn Error+Send+Sync>{e.into()})?;
    let filenames:Vec<String>=dataset.iter().map(|d|io::filename_only(&d.path)).collect();
    io::save_images_to_csv(&filenames,&images,&args.output)?;
    println!("Saved to {:?}",args.output);
    Ok(())
}